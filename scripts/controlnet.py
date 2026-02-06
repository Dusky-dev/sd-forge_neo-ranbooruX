"""
RanbooruX – Forge-Neo ControlNet bridge

Forge-Neo already provides ControlNet as a builtin extension
(sd_forge_controlnet). This file intentionally DOES NOT:

- create UI
- register ControlNet UI callbacks
- duplicate ControlNet logic

It exists only to keep RanbooruX compatible with Forge-Neo
without causing duplicate ControlNet registration or crashes.
"""

import modules.scripts as scripts
from modules import script_callbacks

# Ensure Forge ControlNet API is loaded
from lib_controlnet.api import controlnet_api


class RanbooruXControlNetBridge(scripts.Script):
    """
    Dummy script that defers entirely to Forge's builtin ControlNet.
    """

    sorting_priority = -100  # Load AFTER Forge ControlNet

    def title(self):
        return "RanbooruX ControlNet (Forge)"

    def show(self, is_img2img):
        # Forge-Neo: return False to hide script
        return False

    def ui(self, is_img2img):
        # No UI components
        return ()

    def process(self, p, *args, **kwargs):
        # Forge ControlNet handles everything
        return

    def postprocess(self, p, processed, *args):
        return


# Ensure Forge ControlNet API is initialized
def on_app_started():
    controlnet_api()


script_callbacks.on_app_started(on_app_started)
