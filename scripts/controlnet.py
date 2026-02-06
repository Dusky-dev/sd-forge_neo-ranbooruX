import os
import functools
import tempfile
from typing import Optional, Tuple

import cv2
import torch
import gradio as gr
import numpy as np
from PIL import Image

import modules.scripts as scripts
from modules import shared, script_callbacks, masking, images
from modules.ui_components import InputAccordion
from modules.api.api import decode_base64_to_image
from modules.processing import (
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
)

from lib_controlnet import global_state, external_code
from lib_controlnet.external_code import ControlNetUnit
from lib_controlnet.utils import (
    align_dim_latent,
    set_numpy_seed,
    crop_and_resize_image,
    prepare_mask,
    judge_image_type,
)
from lib_controlnet.controlnet_ui.controlnet_ui_group import ControlNetUiGroup
from lib_controlnet.infotext import Infotext
from lib_controlnet.logging import logger
from lib_controlnet.enums import HiResFixOption
from lib_controlnet.api import controlnet_api

from modules_forge.utils import HWC3, numpy_to_pytorch
from modules_forge.shared import try_load_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher


# -----------------------------------------------------------------------------
# Gradio temp fix
# -----------------------------------------------------------------------------

gradio_tempfile_path = os.path.join(tempfile.gettempdir(), "gradio")
os.makedirs(gradio_tempfile_path, exist_ok=True)

global_state.update_controlnet_filenames()


# -----------------------------------------------------------------------------
# Model cache
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=shared.opts.data.get("control_net_model_cache_size", 5))
def cached_controlnet_loader(filename):
    return try_load_supported_control_model(filename)


# -----------------------------------------------------------------------------
# Cached parameters
# -----------------------------------------------------------------------------

class ControlNetCachedParameters:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.control_cond = None
        self.control_cond_for_hr_fix = None
        self.control_mask = None
        self.control_mask_for_hr_fix = None


# -----------------------------------------------------------------------------
# Main Script
# -----------------------------------------------------------------------------

class ControlNetForForgeOfficial(scripts.Script):
    sorting_priority = 10

    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    # -------------------------------------------------------------------------
    # UI (Photopea fully removed)
    # -------------------------------------------------------------------------

    def ui(self, is_img2img):
        infotext = Infotext()
        ui_groups = []
        controls = []

        max_models = shared.opts.data.get("control_net_unit_count", 3)
        gen_type = "img2img" if is_img2img else "txt2img"
        elem_id_tabname = gen_type + "_controlnet"

        default_unit = ControlNetUnit(
            enabled=False,
            module="None",
            model="None",
        )

        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(
                "ControlNet Integrated",
                open=False,
                elem_id="controlnet",
                elem_classes=["controlnet"],
            ):
                with gr.Row(
                    elem_id=elem_id_tabname + "_accordions",
                    elem_classes="accordions",
                ):
                    for i in range(max_models):
                        with InputAccordion(
                            value=False,
                            label=f"ControlNet Unit {i}",
                            elem_classes=["cnet-unit-enabled-accordion"],
                        ):
                            group = ControlNetUiGroup(is_img2img, default_unit)
                            ui_groups.append(group)
                            controls.append(
                                group.render(f"ControlNet-{i}", elem_id_tabname)
                            )

        for i, ui_group in enumerate(ui_groups):
            infotext.register_unit(i, ui_group)

        if shared.opts.data.get("control_net_sync_field_args", True):
            self.infotext_fields = infotext.infotext_fields
            self.paste_field_names = infotext.paste_field_names

        return tuple(controls)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def get_enabled_units(self, units):
        units = [
            ControlNetUnit.from_dict(u) if isinstance(u, dict) else u
            for u in units
        ]
        return [u for u in units if u.enabled]

    @staticmethod
    def try_crop_image_with_a1111_mask(
        p: StableDiffusionProcessing,
        unit: ControlNetUnit,
        input_image: np.ndarray,
        resize_mode: external_code.ResizeMode,
        preprocessor,
    ) -> np.ndarray:
        a1111_mask = getattr(p, "image_mask", None)
        is_inpaint = (
            isinstance(p, StableDiffusionProcessingImg2Img)
            and p.inpaint_full_res
            and a1111_mask is not None
        )

        if (
            preprocessor.corp_image_with_a1111_mask_when_in_img2img_inpaint_tab
            and is_inpaint
        ):
            mask = prepare_mask(a1111_mask, p)
            crop_region = masking.get_crop_region(
                np.array(mask), p.inpaint_full_res_padding
            )
            crop_region = masking.expand_crop_region(
                crop_region, p.width, p.height, mask.width, mask.height
            )

            input_image = [
                Image.fromarray(input_image[:, :, i])
                for i in range(input_image.shape[2])
            ]
            input_image = [
                images.resize_image(resize_mode.int_value(), i, mask.width, mask.height)
                for i in input_image
            ]
            input_image = [i.crop(crop_region) for i in input_image]
            input_image = [
                images.resize_image(
                    external_code.ResizeMode.OUTER_FIT.int_value(),
                    i,
                    p.width,
                    p.height,
                )
                for i in input_image
            ]
            input_image = [np.asarray(i)[:, :, 0] for i in input_image]
            input_image = np.stack(input_image, axis=2)

        return input_image

    # -------------------------------------------------------------------------
    # Dimensions
    # -------------------------------------------------------------------------

    @staticmethod
    def get_target_dimensions(p: StableDiffusionProcessing) -> Tuple[int, int, int, int]:
        h = align_dim_latent(p.height)
        w = align_dim_latent(p.width)

        if isinstance(p, StableDiffusionProcessingTxt2Img) and getattr(p, "enable_hr", False):
            if p.hr_resize_x == 0 and p.hr_resize_y == 0:
                hr_h = int(p.height * p.hr_scale)
                hr_w = int(p.width * p.hr_scale)
            else:
                hr_h, hr_w = p.hr_resize_y, p.hr_resize_x
            hr_h = align_dim_latent(hr_h)
            hr_w = align_dim_latent(hr_w)
        else:
            hr_h, hr_w = h, w

        return h, w, hr_h, hr_w


# -----------------------------------------------------------------------------
# UI Settings (Photopea options REMOVED)
# -----------------------------------------------------------------------------

def on_ui_settings():
    section = ("control_net", "ControlNet")

    shared.opts.add_option(
        "control_net_models_path",
        shared.OptionInfo("", "Extra ControlNet model path", section=section),
    )
    shared.opts.add_option(
        "control_net_unit_count",
        shared.OptionInfo(
            3,
            "ControlNet unit count (restart required)",
            gr.Slider,
            {"minimum": 1, "maximum": 10, "step": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "control_net_model_cache_size",
        shared.OptionInfo(
            5,
            "ControlNet model cache size",
            gr.Slider,
            {"minimum": 1, "maximum": 10, "step": 1},
            section=section,
        ),
    )
    shared.opts.add_option(
        "control_net_sync_field_args",
        shared.OptionInfo(
            True,
            "Paste ControlNet parameters in infotext",
            gr.Checkbox,
            {"interactive": True},
            section=section,
        ),
    )


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_infotext_pasted(Infotext.on_infotext_pasted)
script_callbacks.on_after_component(ControlNetUiGroup.on_after_component)
script_callbacks.on_before_reload(ControlNetUiGroup.reset)
script_callbacks.on_app_started(controlnet_api)
