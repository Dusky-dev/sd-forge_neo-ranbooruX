import os
from typing import Dict, Optional, Tuple, List, Union

import cv2
import torch

import modules.scripts as scripts
from modules import shared, script_callbacks, masking, images
from modules.ui_components import InputAccordion
from modules.api.api import decode_base64_to_image
import gradio as gr

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
from lib_controlnet.logging import logger
from modules.processing import (
    StableDiffusionProcessingImg2Img,
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessing,
)
from lib_controlnet.infotext import Infotext
from modules_forge.utils import HWC3, numpy_to_pytorch
from lib_controlnet.enums import HiResFixOption
from lib_controlnet.api import controlnet_api

import numpy as np
import functools

from PIL import Image
from modules_forge.shared import try_load_supported_control_model
from modules_forge.supported_controlnet import ControlModelPatcher

# Gradio 3.32 bug fix
import tempfile

gradio_tempfile_path = os.path.join(tempfile.gettempdir(), "gradio")
os.makedirs(gradio_tempfile_path, exist_ok=True)

global_state.update_controlnet_filenames()


@functools.lru_cache(maxsize=shared.opts.data.get("control_net_model_cache_size", 5))
def cached_controlnet_loader(filename):
    return try_load_supported_control_model(filename)


class ControlNetCachedParameters:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.control_cond = None
        self.control_cond_for_hr_fix = None
        self.control_mask = None
        self.control_mask_for_hr_fix = None


class ControlNetForForgeOfficial(scripts.Script):
    sorting_priority = 10

    def title(self):
        return "ControlNet"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        infotext = Infotext()
        ui_groups = []
        controls = []
        max_models = shared.opts.data.get("control_net_unit_count", 3)
        gen_type = "img2img" if is_img2img else "txt2img"
        elem_id_tabname = gen_type + "_controlnet"
        default_unit = ControlNetUnit(enabled=False, module="None", model="None")

        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(
                "ControlNet Integrated",
                open=False,
                elem_id="controlnet",
                elem_classes=["controlnet"],
            ):
                # Photopea completely removed
                photopea = None

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
                            group = ControlNetUiGroup(
                                is_img2img, default_unit, photopea
                            )
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

    # ---- REST OF FILE UNCHANGED ----
