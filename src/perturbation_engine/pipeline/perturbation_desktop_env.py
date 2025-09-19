import logging
import os
from typing import Tuple

from OSWorld.desktop_env.desktop_env import DesktopEnv
from perturbation_engine.control.perturbation_controller import PerturbationController


class PerturbationDesktopEnv(DesktopEnv):
    """Enhanced DesktopEnv that provides perturbation controller"""

    def __init__(
        self,
        provider_name: str = "vmware",
        region: str = None,
        path_to_vm: str = None,
        snapshot_name: str = "init_state",
        action_space: str = "pyautogui",
        cache_dir: str = "cache",
        screen_size: Tuple[int] = (
            int(os.environ.get("SCREEN_WIDTH", 1920)),
            int(os.environ.get("SCREEN_HEIGHT", 1080)),
        ),
        headless: bool = False,
        require_a11y_tree: bool = True,
        require_terminal: bool = False,
        os_type: str = "Ubuntu",
        enable_proxy: bool = False,
        client_password: str = "",
        chromium_port: int = 9222,
    ):
        self.logger = logging.getLogger(__name__)
        super().__init__(
            provider_name=provider_name,
            region=region,
            path_to_vm=path_to_vm,
            snapshot_name=snapshot_name,
            action_space=action_space,
            cache_dir=cache_dir,
            screen_size=screen_size,
            headless=headless,
            require_a11y_tree=require_a11y_tree,
            require_terminal=require_terminal,
            os_type=os_type,
            enable_proxy=enable_proxy,
            client_password=client_password,
        )

        self.logger.info("Perturbation controller initialized")

    def _start_emulator(self):
        """Override to use PerturbationController instead of PythonController"""
        super()._start_emulator()

        # Replace the controller with our enhanced version
        self.controller = PerturbationController(
            vm_ip=self.vm_ip, server_port=self.server_port, chromium_port=self.chromium_port
        )
        self.logger.info("Replaced controller with PerturbationController")

    def reset(self, task_config=None, seed=None, options=None):
        """Override reset to use better debugging"""
        self._log_vm_connection_info()

        try:
            # Setup SSH and auditd if requested in task config
            if task_config and task_config.get("setup_ssh", False):
                self._setup_ssh()

            if task_config and task_config.get("setup_auditd", False):
                self._setup_auditd()

            # Call parent reset with debug controller
            result = super().reset(task_config, seed, options)
            return result
        except Exception as e:
            self.logger.error(f"Reset failed with debug controller: {e}")

            raise

    def _log_vm_connection_info(self):
        """Log VM connection information for SSH access"""
        try:
            # Get VM IP and username
            vm_ip = self.vm_ip
            username = "user"  # Default username for OSWorld VMs

            # Try to get actual username from VM
            try:
                result = self.controller.execute_python_command("import getpass; print(getpass.getuser())")
                if result and result.get("output"):
                    username = result["output"].strip()
            except Exception:
                self.logger.warning("Failed to get username from VM, defaulting to user")

            # Log connection information
            self.logger.info("=" * 60)
            self.logger.info("VM CONNECTION INFORMATION")
            self.logger.info("=" * 60)
            self.logger.info(f"VM IP Address: {vm_ip}")
            self.logger.info(f"Username: {username}")
            self.logger.info(f"SSH Command: ssh {username}@{vm_ip}")
            self.logger.info(f"Server Port: {self.server_port}")
            self.logger.info(f"Chromium Port: {self.chromium_port}")
            self.logger.info("=" * 60)

            # Also print to console for easy copy-paste
            print("\n🔗 VM Connection Info:")
            print(f"   IP: {vm_ip}")
            print(f"   User: {username}")
            print(f"   SSH: ssh {username}@{vm_ip}")
            print()

        except Exception as e:
            self.logger.warning(f"Failed to get VM connection info: {e}")

    def _setup_ssh(self):
        """Setup SSH server in the VM"""
        self.logger.info("Setting up SSH server in VM...")

        try:
            # Install OpenSSH server
            commands = [
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "openssh-server"],
                ["sudo", "systemctl", "enable", "ssh"],
                ["sudo", "systemctl", "start", "ssh"],
                ["sudo", "systemctl", "status", "ssh", "--no-pager"],
            ]

            for cmd in commands:
                result = self.setup_controller._execute_setup(cmd)
                if result and result.get("returncode", 0) != 0:
                    self.logger.warning(f"SSH setup command failed: {' '.join(cmd)}")

            # Configure SSH for easier access
            ssh_config_commands = [
                [
                    "sudo",
                    "sed",
                    "-i",
                    "s/#PasswordAuthentication yes/PasswordAuthentication yes/",
                    "/etc/ssh/sshd_config",
                ],
                [
                    "sudo",
                    "sed",
                    "-i",
                    "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/",
                    "/etc/ssh/sshd_config",
                ],
                ["sudo", "systemctl", "restart", "ssh"],
            ]

            for cmd in ssh_config_commands:
                self.setup_controller._execute_setup(cmd)

            self.logger.info("SSH server setup completed")

        except Exception as e:
            self.logger.error(f"SSH setup failed: {e}")

    def _setup_auditd(self):
        """Setup auditd for system auditing in the VM"""
        self.logger.info("Setting up auditd in VM...")

        try:
            # Install auditd
            commands = [
                ["sudo", "apt-get", "update"],
                ["sudo", "apt-get", "install", "-y", "auditd", "audispd-plugins"],
                ["sudo", "systemctl", "enable", "auditd"],
                ["sudo", "systemctl", "start", "auditd"],
                ["sudo", "systemctl", "status", "auditd", "--no-pager"],
            ]

            for cmd in commands:
                result = self.setup_controller._execute_setup(cmd)
                if result and result.get("returncode", 0) != 0:
                    self.logger.warning(f"Auditd setup command failed: {' '.join(cmd)}")

            # Configure auditd for comprehensive logging
            audit_rules = [
                "-w /home/user -p rwxa -k user_activity",
                "-w /tmp -p rwxa -k temp_activity",
                "-w /var/log -p rwxa -k log_activity",
                "-a always,exit -F arch=b64 -S execve -k process_execution",
                "-a always,exit -F arch=b32 -S execve -k process_execution",
            ]

            for rule in audit_rules:
                self.setup_controller._execute_setup(
                    ["sudo", "auditctl", "-w", rule.split()[1], "-p", rule.split()[3], "-k", rule.split()[5]]
                )

            self.logger.info("Auditd setup completed")

        except Exception as e:
            self.logger.error(f"Auditd setup failed: {e}")

    def close(self) -> None:
        """Close both the perturbation controller and original environment"""
        self.controller.close_playwright()
        super().close()
