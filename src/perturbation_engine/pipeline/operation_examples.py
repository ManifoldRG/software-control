"""
Operation Examples: Centralized concrete operation examples for LLM prompts
"""

from typing import List


class OperationExamples:
    """Centralized operation examples for different applications"""

    @staticmethod
    def get_vlc_examples() -> List[str]:
        """Get VLC-specific operation examples"""
        return [
            'VLC Theme Change: execute_python_command(\'VLCTools.set_settings("qt-theme", "dark")\')',
            'VLC Volume Control: execute_python_command(\'VLCTools.set_settings("volume", "200")\')',
            "VLC Window Resize: execute_python_command('VLCTools.set_window_size(800, 600)')",
            "VLC Video Filter: execute_vlc_visual_effects('apply_video_filter_blur', {'target_app': 'vlc'})",
            "VLC Aspect Ratio: execute_vlc_visual_effects('change_aspect_ratio_16_9', {'target_app': 'vlc'})",
            "System Theme for VLC: execute_system_theme_coherence({'target_app': 'vlc'})",
        ]

    @staticmethod
    def get_chrome_examples() -> List[str]:
        """Get Chrome-specific operation examples"""
        return [
            "Chrome CSS Injection: execute_css_injection('body { background-color: #ff0000 !important; }', {'target_app': 'chrome'})",
            "Chrome Theme Change: execute_python_command('BrowserTools.set_theme(\"dark\")')",
            "Chrome Tab Management: execute_python_command('BrowserTools.open_tab(\"https://example.com\")')",
            "Chrome DOM Modification: execute_dom_modification('document.body.style.filter = \"hue-rotate(180deg)\"', {'target_app': 'chrome'})",
            "Chrome Visual Manipulation: execute_chrome_visual_manipulation('inject_custom_css_red_theme', {'target_app': 'chrome'})",
            "System Theme for Chrome: execute_system_theme_coherence({'target_app': 'chrome'})",
        ]

    @staticmethod
    def get_calc_examples() -> List[str]:
        """Get LibreOffice Calc-specific operation examples"""
        return [
            "Calc Theme Change: execute_uno_command('CalcTools.set_theme(\"dark\")', {'target_app': 'calc'})",
            "Calc Cell Formatting: execute_uno_command('CalcTools.format_range(\"A1:C10\", \"background_color\", \"#ff0000\")', {'target_app': 'calc'})",
            "Calc Window Management: execute_uno_command('CalcTools.set_window_size(1000, 800)', {'target_app': 'calc'})",
            "Calc Visual Formatting: execute_libreoffice_visual_formatting('randomize_cell_colors', {'target_app': 'calc'})",
            "System Theme for Calc: execute_system_theme_coherence({'target_app': 'calc'})",
        ]

    @staticmethod
    def get_writer_examples() -> List[str]:
        """Get LibreOffice Writer-specific operation examples"""
        return [
            "Writer Theme Change: execute_uno_command('WriterTools.set_theme(\"dark\")', {'target_app': 'writer'})",
            "Writer Font Modification: execute_uno_command('WriterTools.set_font(\"Arial\", 14)', {'target_app': 'writer'})",
            "Writer Text Formatting: execute_uno_command('WriterTools.set_color(\"#ff0000\")', {'target_app': 'writer'})",
            "Writer Visual Formatting: execute_libreoffice_visual_formatting('change_font_rendering', {'target_app': 'writer'})",
            "System Theme for Writer: execute_system_theme_coherence({'target_app': 'writer'})",
        ]

    @staticmethod
    def get_code_examples() -> List[str]:
        """Get Code editor-specific operation examples"""
        return [
            "Code Theme Change: execute_python_command('CodeTools.set_theme(\"dark\")')",
            "Code Font Size: execute_python_command('CodeTools.set_font_size(16)')",
            "Code Window Layout: execute_python_command('CodeTools.toggle_sidebar()')",
            "System Theme for Code: execute_system_theme_coherence({'target_app': 'code'})",
        ]

    @staticmethod
    def get_system_examples() -> List[str]:
        """Get universal system-level operation examples"""
        return [
            "Universal System Theme: execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
            "Universal Font Change: execute_bash_command('gsettings set org.gnome.desktop.interface font-name \"Liberation Sans 14\"')",
            "Universal Wallpaper: execute_bash_command('gsettings set org.gnome.desktop.background picture-uri \"file:///usr/share/backgrounds/ubuntu-mate-photos/ubuntu-mate-dark.jpg\"')",
            "Universal Cursor Theme: execute_bash_command('gsettings set org.gnome.desktop.interface cursor-theme \"Adwaita\"')",
        ]

    @staticmethod
    def get_examples_for_app(app_name: str) -> List[str]:
        """Get examples for a specific app"""
        app_lower = app_name.lower()

        if app_lower == "vlc":
            return OperationExamples.get_vlc_examples()
        elif app_lower in ["chrome", "google_chrome"]:
            return OperationExamples.get_chrome_examples()
        elif app_lower in ["libreoffice_calc", "calc"]:
            return OperationExamples.get_calc_examples()
        elif app_lower in ["libreoffice_writer", "writer"]:
            return OperationExamples.get_writer_examples()
        elif app_lower in ["code", "vscode"]:
            return OperationExamples.get_code_examples()
        else:
            # System-level perturbations for unknown apps
            return [
                f"System Theme for {app_name}: execute_system_theme_coherence({{'target_app': '{app_name}'}})",
                f"Desktop Environment for {app_name}: execute_bash_command('gsettings set org.gnome.desktop.interface gtk-theme \"Adwaita-dark\"')",
                f"System Font for {app_name}: execute_bash_command('gsettings set org.gnome.desktop.interface font-name \"Ubuntu 12\"')",
            ]

    @staticmethod
    def get_all_examples_for_apps(app_names: List[str]) -> List[str]:
        """Get all examples for a list of apps"""
        examples = []

        for app in app_names:
            examples.extend(OperationExamples.get_examples_for_app(app))

        # Add universal system-level examples
        examples.extend(OperationExamples.get_system_examples())

        return examples
