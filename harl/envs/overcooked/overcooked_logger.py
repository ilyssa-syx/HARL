"""Logger for HARL Overcooked experiments."""

from harl.common.base_logger import BaseLogger


class OvercookedLogger(BaseLogger):
    """Use the layout name as the Overcooked task identifier."""

    def get_task_name(self):
        return self.env_args["layout_name"]
