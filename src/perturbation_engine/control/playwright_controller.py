from OSWorld.desktop_env.controllers.python import PythonController


class PlaywrightController(PythonController):
    def __init__(self, vm_ip: str, server_port: int):
        super().__init__(vm_ip, server_port)

    def get_page(self):
        return self.controller.page
