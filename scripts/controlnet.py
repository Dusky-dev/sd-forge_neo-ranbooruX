"""
RanbooruX – Forge-Neo ControlNet bridge

Forge-Neo already provides ControlNet as a builtin extension
(sd_forge_controlnet). This file intentionally DOES NOT:

- create UI
- register callbacks
- duplicate ControlNet logic
- touch ControlNetUiGroup

It exists only to keep RanbooruX compatible with Forge-Neo
without crashing due to duplicate ControlNet registration.
"""

import modules.scripts as scripts
from modules import script_callbacks

# Import Forge's ControlNet API to ensure it is loaded
from lib_controlnet.api import controlnet_api


class RanbooruXControlNetBridge(scripts.Script):
    """
    Dummy script that defers entirely to Forge's builtin ControlNet.
    """

    sorting_priority = -100  # Load AFTER Forge ControlNet

    def title(self):
        # Do not expose a second ControlNet UI
        return "RanbooruX ControlNet (Forge)"

    def show(self, is_img2img):
        # Always hidden – Forge already provides the UI
        return scripts.NeverVisible

    def ui(self, is_img2img):
        # No UI components
        return ()

    def process(self, p, *args, **kwargs):
        # No processing – Forge ControlNet handles everything
        return

    def postprocess(self, p, processed, *args):
        return


# Ensure Forge ControlNet API is initialized
def on_app_started():
    controlnet_api()


script_callbacks.on_app_started(on_app_started)
