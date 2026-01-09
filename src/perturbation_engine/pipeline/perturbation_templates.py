"""
Refactored Perturbation System with Predefined Templates

This module replaces the complex OperationCatalog with a clean, maintainable
system of predefined perturbation templates per app type.

Key Design Principles:
- Predefined templates for each app type
- Runtime parameter resolution
- Structured LLM calls with Pydantic models
- Clean separation of concerns
- Built-in safety constraints
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Import existing models and utilities
from .data_models import (
    ApiCallType,
    ExecutionContext,
    PerturbationCategory,
    PerturbationIntensity,
    PerturbationType,
    ScenarioSpec,
    TemplateCategory,
)
from .llm_services import BaseLLM


@dataclass
class PerturbationTemplate:
    """Template for perturbation operations - optimized for actual command complexity"""

    name: str
    category: TemplateCategory
    description: str
    api_call: ApiCallType
    template_command: str  # Single-line command with {param} placeholders
    parameters: Dict[str, Any] = field(default_factory=dict)  # Default parameter values
    target_elements: List[str] = field(default_factory=list)  # Target element types
    safety_constraints: List[str] = field(default_factory=list)  # Safety constraints
    educational_value: str = ""  # Educational rationale
    risk_level: str = "low"  # Risk level


class PerturbationParameters(BaseModel):
    """Parameters for perturbation commands - optimized for actual template usage"""

    # Chrome/DOM parameters
    selector: str = ""  # CSS selector: "button", "h1", ".menu"
    color: str = ""  # Hex color: "#ff0000", "#cccccc"
    text: str = ""  # Text content: "Modified Text", "New Label"
    radius: str = ""  # Border radius: "12px", "4px"
    shadow: str = ""  # Box shadow: "0 4px 8px rgba(0,0,0,0.2)"
    container: str = ""  # Container selector: ".menu", "#main-content"
    direction: str = ""  # Flex direction: "column-reverse", "row"
    justify: str = ""  # Justify content: "center", "space-between"
    align: str = ""  # Align items: "center", "flex-start"
    gap: str = ""  # Gap between elements: "10px", "1rem"

    # System parameters
    theme: str = ""  # Theme name: "Adwaita-dark", "Papirus-Dark"
    font: str = ""  # Font name: "Ubuntu 12", "Liberation Sans 14"
    title: str = ""  # Notification title: "System Alert"
    message: str = ""  # Notification message: "Theme change applied"

    # File system parameters
    old_name: str = ""  # Original filename: "test.py", "main.ods"
    new_name: str = ""  # New filename: "modified_test.py", "updated_main.ods"
    folder: str = ""  # Folder path: "temp", "src"
    file: str = ""  # File to create: "new_file.txt", "temp.py"

    # LibreOffice parameters
    col: str = ""  # Column index: "0", "1", "2"
    row: str = ""  # Row index: "0", "1", "2"
    formula: str = ""  # Cell formula: "=1+1", "=SUM(A1:A10)"
    weight: str = ""  # Font weight: "150", "bold"

    # Window management parameters
    x: str = ""  # X position: "100", "200"
    y: str = ""  # Y position: "100", "200"
    width: str = ""  # Window width: "1200", "800"
    height: str = ""  # Window height: "800", "600"
    window_name: str = ""  # Window name: "Code", "Calculator"


class PerturbationDecision(BaseModel):
    """Structured decision for perturbation application"""

    should_apply: bool
    reasoning: str
    template_name: str
    api_call: str
    generated_command: str
    parameters: PerturbationParameters
    confidence: float
    intensity: str = "medium"  # Intensity level for the perturbation
    alternative_commands: List[str] = Field(default_factory=list)
    visual_impact: str = ""
    coherence_rationale: str = ""


class AppPerturbationTemplates:
    """Predefined perturbation templates organized by app type"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates = self._build_app_templates()

    def _build_app_templates(self) -> Dict[str, List[PerturbationTemplate]]:
        """Build predefined templates for each app type"""
        return {
            "chrome": self._get_chrome_templates(),
            "chromium": self._get_chrome_templates(),
            "google-chrome": self._get_chrome_templates(),
            "vscode": self._get_vscode_templates(),
            "code": self._get_vscode_templates(),
            "os": self._get_os_templates(),
            "system": self._get_os_templates(),
            "libreoffice_calc": self._get_libreoffice_calc_templates(),
            "calc": self._get_libreoffice_calc_templates(),
            "libreoffice_writer": self._get_libreoffice_writer_templates(),
            "writer": self._get_libreoffice_writer_templates(),
            "libreoffice_impress": self._get_libreoffice_impress_templates(),
            "impress": self._get_libreoffice_impress_templates(),
        }

    def _get_chrome_templates(self) -> List[PerturbationTemplate]:
        """Chrome-specific perturbation templates - DRAMATIC VISUAL CHANGES"""
        return [
            PerturbationTemplate(
                name="dramatic_color_inversion",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Invert all colors on the page",
                api_call=ApiCallType.EXECUTE_CSS_INJECTION,
                template_command="document.body.style.filter = 'invert(1) hue-rotate(180deg)'; document.body.style.backgroundColor = '{bg_color}'",
                parameters={"bg_color": "#000000"},
                target_elements=["body", "all"],
                safety_constraints=["Visual only", "Reversible"],
                educational_value="Adapt to completely inverted color schemes",
            ),
            PerturbationTemplate(
                name="massive_font_change",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Change all text to massive, bold fonts",
                api_call=ApiCallType.EXECUTE_CSS_INJECTION,
                template_command="document.querySelectorAll('*').forEach(el => {{ el.style.fontSize = '{size}'; el.style.fontWeight = 'bold'; el.style.fontFamily = '{font}'; }})",
                parameters={"size": "24px", "font": "Impact, Arial Black, sans-serif"},
                target_elements=["all"],
                safety_constraints=["Visual only", "No functionality changes"],
                educational_value="Adapt to dramatically different typography",
            ),
            PerturbationTemplate(
                name="rainbow_background",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Add rainbow gradient background",
                api_call=ApiCallType.EXECUTE_CSS_INJECTION,
                template_command="document.body.style.background = 'linear-gradient(45deg, #ff0000, #ff8000, #ffff00, #80ff00, #00ff00, #00ff80, #00ffff, #0080ff, #0000ff, #8000ff, #ff00ff, #ff0080)'; document.body.style.backgroundSize = '400% 400%'; document.body.style.animation = 'rainbow 3s ease infinite'",
                parameters={},
                target_elements=["body"],
                safety_constraints=["Visual only", "Eye-catching"],
                educational_value="Adapt to highly distracting visual environments",
            ),
            PerturbationTemplate(
                name="element_rotation",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Rotate all elements randomly",
                api_call=ApiCallType.EXECUTE_CSS_INJECTION,
                template_command="document.querySelectorAll('{selector}').forEach(el => {{ el.style.transform = 'rotate({angle}deg)'; el.style.transition = 'transform 0.5s ease'; }})",
                parameters={"selector": "button, input, div", "angle": "15"},
                target_elements=["button", "input", "div"],
                safety_constraints=["Visual only", "No functionality changes"],
                educational_value="Adapt to rotated UI elements",
            ),
            PerturbationTemplate(
                name="blinking_elements",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Make elements blink/flash",
                api_call=ApiCallType.EXECUTE_CSS_INJECTION,
                template_command="document.querySelectorAll('{selector}').forEach(el => {{ el.style.animation = 'blink 1s infinite'; }})",
                parameters={"selector": "button, a, h1, h2"},
                target_elements=["button", "link", "heading"],
                safety_constraints=["Visual only", "Attention-grabbing"],
                educational_value="Adapt to flashing/blinking UI elements",
            ),
            PerturbationTemplate(
                name="system_gtk_theme",
                category=TemplateCategory.SYSTEM,
                description="Change system GTK theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface gtk-theme '{theme}'",
                parameters={"theme": "Adwaita-dark"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Learn to work with different system themes",
            ),
            PerturbationTemplate(
                name="system_fonts",
                category=TemplateCategory.SYSTEM,
                description="Change system font settings",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface font-name '{font}'",
                parameters={"font": "Ubuntu 12"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to different font rendering",
            ),
            PerturbationTemplate(
                name="icon_theme",
                category=TemplateCategory.SYSTEM,
                description="Change system icon theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface icon-theme '{theme}'",
                parameters={"theme": "Papirus-Dark"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Recognize elements with different icon styles",
            ),
            PerturbationTemplate(
                name="system_notifications",
                category=TemplateCategory.NOTIFICATION,
                description="Show system notifications",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="notify-send '{title}' '{message}'",
                parameters={"title": "System Alert", "message": "Visual change applied"},
                safety_constraints=["Non-blocking", "Temporary"],
                educational_value="Handle visual distractions and notifications",
            ),
            PerturbationTemplate(
                name="app_theme",
                category=TemplateCategory.THEME,
                description="Change Chrome app theme",
                api_call=ApiCallType.EXECUTE_JS_ON_PAGE,
                template_command="chrome.theme.update({{'images': {{'theme_frame': '{image_url}'}}, 'colors': {{'frame': '{color}'}}}})",
                parameters={"image_url": "", "color": "#1a1a1a"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different app themes",
            ),
            PerturbationTemplate(
                name="app_fonts",
                category=TemplateCategory.THEME,
                description="Change Chrome font settings",
                api_call=ApiCallType.EXECUTE_JS_ON_PAGE,
                template_command="chrome.fontSettings.setFont({{'genericFamily': 'serif', 'script': 'latin', 'fontId': '{font_id}'}})",
                parameters={"font_id": "Times New Roman"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different font rendering in apps",
            ),
        ]

    def _get_vscode_templates(self) -> List[PerturbationTemplate]:
        """VSCode-specific perturbation templates - DRAMATIC DEVELOPMENT ENVIRONMENT CHANGES"""
        return [
            PerturbationTemplate(
                name="dramatic_theme_change",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Switch to high-contrast system theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface gtk-theme '{theme}' && gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'",
                parameters={"theme": "HighContrast"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to high-contrast accessibility themes",
            ),
            PerturbationTemplate(
                name="massive_monospace_font",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Change to massive monospace fonts",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface font-name '{font}' && gsettings set org.gnome.desktop.interface monospace-font-name '{mono_font}'",
                parameters={"font": "Liberation Sans 18", "mono_font": "Liberation Mono 20"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to large accessibility fonts in development",
            ),
            PerturbationTemplate(
                name="persistent_dev_notifications",
                category=TemplateCategory.NOTIFICATION,
                description="DRAMATIC: Send multiple development notifications",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="for i in {{1..3}}; do notify-send -u critical 'DEV ALERT $i' 'Code compilation failed - Check syntax!'; sleep 0.5; done",
                parameters={},
                safety_constraints=["Multiple notifications", "High priority"],
                educational_value="Handle development-related notification spam",
            ),
            PerturbationTemplate(
                name="confusing_file_names",
                category=TemplateCategory.FILE_SYSTEM,
                description="DRAMATIC: Rename files to confusing names",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="mv '{old_name}' '{new_name}'",
                parameters={"old_name": "main.py", "new_name": "CONFUSING_MAIN_FILE_123.py"},
                safety_constraints=["Workspace only", "No critical files"],
                educational_value="Adapt to confusing file naming conventions",
            ),
            PerturbationTemplate(
                name="workspace_clutter",
                category=TemplateCategory.FILE_SYSTEM,
                description="DRAMATIC: Add many confusing files to workspace",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="for i in {{1..5}}; do touch '{folder}/confusing_file_$i.txt'; done",
                parameters={"folder": "src"},
                safety_constraints=["Temporary files only", "No source code"],
                educational_value="Handle cluttered development environments",
            ),
            PerturbationTemplate(
                name="window_repositioning",
                category=TemplateCategory.WINDOW_MANAGEMENT,
                description="DRAMATIC: Move VSCode window to unexpected position",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="wmctrl -r 'Code' -e 0,{x},{y},-1,-1",
                parameters={"x": 50, "y": 50},
                safety_constraints=["Reposition only", "No resizing"],
                educational_value="Adapt to unexpected window positions",
            ),
        ]

    def _get_os_templates(self) -> List[PerturbationTemplate]:
        """OS/System-specific perturbation templates - DRAMATIC SYSTEM CHANGES"""
        return [
            PerturbationTemplate(
                name="dramatic_theme_change",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Switch to high-contrast theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface gtk-theme '{theme}' && gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'",
                parameters={"theme": "HighContrast"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to high-contrast accessibility themes",
            ),
            PerturbationTemplate(
                name="massive_font_change",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Change to massive system fonts",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface font-name '{font}' && gsettings set org.gnome.desktop.interface document-font-name '{font}' && gsettings set org.gnome.desktop.interface monospace-font-name '{mono_font}'",
                parameters={"font": "Liberation Sans 18", "mono_font": "Liberation Mono 18"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to large accessibility fonts",
            ),
            PerturbationTemplate(
                name="dramatic_icon_change",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Change to completely different icon theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface icon-theme '{theme}'",
                parameters={"theme": "HighContrast"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to accessibility icon themes",
            ),
            PerturbationTemplate(
                name="persistent_notification_spam",
                category=TemplateCategory.NOTIFICATION,
                description="DRAMATIC: Send multiple persistent notifications",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="for i in {{1..3}}; do notify-send -u critical '{title}' '{message} $i'; sleep 0.5; done",
                parameters={"title": "URGENT SYSTEM ALERT", "message": "Critical notification"},
                safety_constraints=["Multiple notifications", "High priority"],
                educational_value="Handle notification spam and distractions",
            ),
            PerturbationTemplate(
                name="desktop_wallpaper_change",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Change desktop wallpaper to solid bright color",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.background picture-uri 'file:///usr/share/pixmaps/backgrounds/gnome/{color}.jpg' || gsettings set org.gnome.desktop.background picture-uri ''",
                parameters={"color": "bright"},
                safety_constraints=["Visual only", "Reversible"],
                educational_value="Adapt to distracting desktop backgrounds",
            ),
            PerturbationTemplate(
                name="file_name",
                category=TemplateCategory.FILE_SYSTEM,
                description="Rename files",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="mv '{old_name}' '{new_name}'",
                parameters={"old_name": "test.txt", "new_name": "modified_test.txt"},
                safety_constraints=["Temporary files only", "No critical files"],
                educational_value="Adapt to changing file names",
            ),
            PerturbationTemplate(
                name="folder_content",
                category=TemplateCategory.FILE_SYSTEM,
                description="Modify folder contents",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="touch '{folder}/{file}'",
                parameters={"folder": "temp", "file": "new_file.txt"},
                safety_constraints=["Temporary files only", "No source code"],
                educational_value="Handle changing file structure",
            ),
            PerturbationTemplate(
                name="window_management",
                category=TemplateCategory.WINDOW_MANAGEMENT,
                description="Reposition windows (no resizing)",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="wmctrl -r '{window_name}' -e 0,{x},{y},-1,-1",
                parameters={"window_name": "Calculator", "x": 200, "y": 200},
                safety_constraints=["Reposition only", "No resizing", "No critical windows"],
                educational_value="Adapt to different window positions",
            ),
        ]

    def _get_libreoffice_calc_templates(self) -> List[PerturbationTemplate]:
        """LibreOffice Calc-specific perturbation templates - DRAMATIC SPREADSHEET CHANGES"""
        return [
            PerturbationTemplate(
                name="dramatic_cell_colors",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Fill entire sheet with bright colors",
                api_call=ApiCallType.EXECUTE_UNO_COMMAND,
                template_command="import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx); desktop = resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext').ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')); doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); range = sheet.getCellRangeByName('A1:Z20'); range.CellBackColor = {color}",
                parameters={"color": "0xFF0000"},
                target_elements=["cell_range"],
                safety_constraints=["Visual only", "Large area"],
                educational_value="Adapt to dramatically colored spreadsheets",
            ),
            PerturbationTemplate(
                name="massive_font_change",
                category=TemplateCategory.VISUAL,
                description="DRAMATIC: Change all text to massive fonts",
                api_call=ApiCallType.EXECUTE_UNO_COMMAND,
                template_command="import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx); desktop = resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext').ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')); doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); range = sheet.getCellRangeByName('A1:Z20'); range.CharHeight = {size}; range.CharWeight = 150",
                parameters={"size": "24"},
                target_elements=["cell_range"],
                safety_constraints=["Visual only", "Large area"],
                educational_value="Adapt to oversized text in spreadsheets",
            ),
            PerturbationTemplate(
                name="sheet_name_change",
                category=TemplateCategory.CONTENT,
                description="DRAMATIC: Rename sheet to confusing name",
                api_call=ApiCallType.EXECUTE_UNO_COMMAND,
                template_command="import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx); desktop = resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext').ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')); doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); sheet.setName('{new_name}')",
                parameters={"new_name": "CONFUSING_SHEET_NAME_123"},
                target_elements=["sheet"],
                safety_constraints=["Non-critical sheets only"],
                educational_value="Adapt to confusing sheet names",
            ),
            PerturbationTemplate(
                name="dramatic_system_theme",
                category=TemplateCategory.SYSTEM,
                description="DRAMATIC: Switch to high-contrast system theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface gtk-theme '{theme}' && gsettings set org.gnome.desktop.interface color-scheme 'prefer-dark'",
                parameters={"theme": "HighContrast"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to high-contrast accessibility themes",
            ),
            PerturbationTemplate(
                name="persistent_calc_notifications",
                category=TemplateCategory.NOTIFICATION,
                description="DRAMATIC: Send multiple Calc-specific notifications",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="for i in {{1..3}}; do notify-send -u critical 'CALC ALERT $i' 'Spreadsheet modified - Check your data!'; sleep 0.5; done",
                parameters={},
                safety_constraints=["Multiple notifications", "High priority"],
                educational_value="Handle spreadsheet-related notification spam",
            ),
            PerturbationTemplate(
                name="background_sheet_name",
                category=TemplateCategory.CONTENT,
                description="Change sheet name",
                api_call=ApiCallType.EXECUTE_UNO_COMMAND,
                template_command="import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx); desktop = resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext').ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')); doc = desktop.getCurrentComponent(); sheet = doc.getSheets().getByIndex(0); sheet.setName('{new_name}')",
                parameters={"new_name": "Modified Sheet"},
                safety_constraints=["Non-critical sheets only"],
                educational_value="Adapt to changing sheet names",
            ),
            PerturbationTemplate(
                name="app_theme",
                category=TemplateCategory.THEME,
                description="Change LibreOffice theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.libreoffice.LibreOffice Calc '{setting}' '{value}'",
                parameters={"setting": "theme", "value": "dark"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different app themes",
            ),
            PerturbationTemplate(
                name="app_fonts",
                category=TemplateCategory.THEME,
                description="Change LibreOffice font settings",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.libreoffice.LibreOffice Calc '{setting}' '{value}'",
                parameters={"setting": "font", "value": "Liberation Sans"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different font rendering in apps",
            ),
            PerturbationTemplate(
                name="file_name",
                category=TemplateCategory.FILE_SYSTEM,
                description="Rename Calc file",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="mv '{old_name}' '{new_name}'",
                parameters={"old_name": "test.ods", "new_name": "modified_test.ods"},
                safety_constraints=["Temporary files only", "No critical files"],
                educational_value="Adapt to changing file names",
            ),
            PerturbationTemplate(
                name="folder_content",
                category=TemplateCategory.FILE_SYSTEM,
                description="Modify folder contents",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="touch '{folder}/{file}'",
                parameters={"folder": "temp", "file": "new_file.ods"},
                safety_constraints=["Temporary files only", "No source code"],
                educational_value="Handle changing file structure",
            ),
        ]

    def _get_libreoffice_writer_templates(self) -> List[PerturbationTemplate]:
        """LibreOffice Writer-specific perturbation templates"""
        return [
            PerturbationTemplate(
                name="system_gtk_theme",
                category=TemplateCategory.SYSTEM,
                description="Change system GTK theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface gtk-theme '{theme}'",
                parameters={"theme": "Adwaita-dark"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Learn to work with different system themes",
            ),
            PerturbationTemplate(
                name="system_fonts",
                category=TemplateCategory.SYSTEM,
                description="Change system font settings",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface font-name '{font}'",
                parameters={"font": "Liberation Serif 12"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to different font rendering",
            ),
            PerturbationTemplate(
                name="icon_theme",
                category=TemplateCategory.SYSTEM,
                description="Change system icon theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface icon-theme '{theme}'",
                parameters={"theme": "Papirus-Dark"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Recognize elements with different icon styles",
            ),
            PerturbationTemplate(
                name="system_notifications",
                category=TemplateCategory.NOTIFICATION,
                description="Show system notifications",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="notify-send '{title}' '{message}'",
                parameters={"title": "Writer Alert", "message": "Formatting change applied"},
                safety_constraints=["Non-blocking", "Temporary"],
                educational_value="Handle visual distractions and notifications",
            ),
            PerturbationTemplate(
                name="text_formatting",
                category=TemplateCategory.VISUAL,
                description="Change text formatting",
                api_call=ApiCallType.EXECUTE_UNO_COMMAND,
                template_command="import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx); desktop = resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext').ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')); doc = desktop.getCurrentComponent(); text = doc.getText(); cursor = text.createTextCursor(); cursor.CharWeight = {weight}; text.insertString(cursor, '{text}', False)",
                parameters={"weight": "150", "text": "Modified Text"},
                target_elements=["text"],
                safety_constraints=["Visual only", "No content changes"],
                educational_value="Recognize text with different formatting",
            ),
            PerturbationTemplate(
                name="app_theme",
                category=TemplateCategory.THEME,
                description="Change LibreOffice theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.libreoffice.LibreOffice Writer '{setting}' '{value}'",
                parameters={"setting": "theme", "value": "dark"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different app themes",
            ),
            PerturbationTemplate(
                name="app_fonts",
                category=TemplateCategory.THEME,
                description="Change LibreOffice font settings",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.libreoffice.LibreOffice Writer '{setting}' '{value}'",
                parameters={"setting": "font", "value": "Liberation Serif"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different font rendering in apps",
            ),
            PerturbationTemplate(
                name="file_name",
                category=TemplateCategory.FILE_SYSTEM,
                description="Rename Writer file",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="mv '{old_name}' '{new_name}'",
                parameters={"old_name": "test.odt", "new_name": "modified_test.odt"},
                safety_constraints=["Temporary files only", "No critical files"],
                educational_value="Adapt to changing file names",
            ),
            PerturbationTemplate(
                name="folder_content",
                category=TemplateCategory.FILE_SYSTEM,
                description="Modify folder contents",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="touch '{folder}/{file}'",
                parameters={"folder": "temp", "file": "new_file.odt"},
                safety_constraints=["Temporary files only", "No source code"],
                educational_value="Handle changing file structure",
            ),
        ]

    def _get_libreoffice_impress_templates(self) -> List[PerturbationTemplate]:
        """LibreOffice Impress-specific perturbation templates"""
        return [
            PerturbationTemplate(
                name="system_gtk_theme",
                category=TemplateCategory.SYSTEM,
                description="Change system GTK theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface gtk-theme '{theme}'",
                parameters={"theme": "Adwaita-dark"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Learn to work with different system themes",
            ),
            PerturbationTemplate(
                name="system_fonts",
                category=TemplateCategory.SYSTEM,
                description="Change system font settings",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface font-name '{font}'",
                parameters={"font": "Liberation Sans 12"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Adapt to different font rendering",
            ),
            PerturbationTemplate(
                name="icon_theme",
                category=TemplateCategory.SYSTEM,
                description="Change system icon theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.gnome.desktop.interface icon-theme '{theme}'",
                parameters={"theme": "Papirus-Dark"},
                safety_constraints=["System-wide change", "Reversible"],
                educational_value="Recognize elements with different icon styles",
            ),
            PerturbationTemplate(
                name="system_notifications",
                category=TemplateCategory.NOTIFICATION,
                description="Show system notifications",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="notify-send '{title}' '{message}'",
                parameters={"title": "Impress Alert", "message": "Slide change applied"},
                safety_constraints=["Non-blocking", "Temporary"],
                educational_value="Handle visual distractions and notifications",
            ),
            PerturbationTemplate(
                name="slide_background",
                category=TemplateCategory.VISUAL,
                description="Change slide background",
                api_call=ApiCallType.EXECUTE_UNO_COMMAND,
                template_command="import uno; ctx = uno.getComponentContext(); resolver = ctx.ServiceManager.createInstanceWithContext('com.sun.star.bridge.UnoUrlResolver', ctx); desktop = resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext').ServiceManager.createInstanceWithContext('com.sun.star.frame.Desktop', resolver.resolve('uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')); doc = desktop.getCurrentComponent(); pages = doc.getDrawPages(); page = pages.getByIndex(0); page.FillColor = {color}",
                parameters={"color": "0xFF0000"},
                target_elements=["slide"],
                safety_constraints=["Visual only", "No content changes"],
                educational_value="Recognize slides with different backgrounds",
            ),
            PerturbationTemplate(
                name="app_theme",
                category=TemplateCategory.THEME,
                description="Change LibreOffice theme",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.libreoffice.LibreOffice Impress '{setting}' '{value}'",
                parameters={"setting": "theme", "value": "dark"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different app themes",
            ),
            PerturbationTemplate(
                name="app_fonts",
                category=TemplateCategory.THEME,
                description="Change LibreOffice font settings",
                api_call=ApiCallType.EXECUTE_BASH_COMMAND,
                template_command="gsettings set org.libreoffice.LibreOffice Impress '{setting}' '{value}'",
                parameters={"setting": "font", "value": "Liberation Sans"},
                safety_constraints=["App-specific", "Reversible"],
                educational_value="Adapt to different font rendering in apps",
            ),
            PerturbationTemplate(
                name="file_name",
                category=TemplateCategory.FILE_SYSTEM,
                description="Rename Impress file",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="mv '{old_name}' '{new_name}'",
                parameters={"old_name": "test.odp", "new_name": "modified_test.odp"},
                safety_constraints=["Temporary files only", "No critical files"],
                educational_value="Adapt to changing file names",
            ),
            PerturbationTemplate(
                name="folder_content",
                category=TemplateCategory.FILE_SYSTEM,
                description="Modify folder contents",
                api_call=ApiCallType.EXECUTE_FILE_SYSTEM_MANIPULATION,
                template_command="touch '{folder}/{file}'",
                parameters={"folder": "temp", "file": "new_file.odp"},
                safety_constraints=["Temporary files only", "No source code"],
                educational_value="Handle changing file structure",
            ),
        ]

    def get_templates_for_app(self, app_name: str) -> List[PerturbationTemplate]:
        """Get predefined templates for specific app"""
        app_name_lower = app_name.lower()

        # Try exact match first
        if app_name_lower in self.templates:
            return self.templates[app_name_lower]

        # Try partial matches
        for template_key, templates in self.templates.items():
            if app_name_lower in template_key or template_key in app_name_lower:
                return templates

        # Fallback to system templates
        return self.templates.get("system", [])

    def get_template_by_name(self, app_name: str, template_name: str) -> Optional[PerturbationTemplate]:
        """Get specific template by name for app"""
        templates = self.get_templates_for_app(app_name)
        for template in templates:
            if template.name == template_name:
                return template
        return None


class TemplateBasedPerturbationGenerator(BaseLLM):
    """Simplified perturbation generator using predefined templates"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", model_provider: str = "gemini"):
        super().__init__(model_name, model_provider)
        self.templates = AppPerturbationTemplates()
        self.logger = logging.getLogger(__name__)

    def decide_perturbation(
        self, execution_context: ExecutionContext, scenario_spec: ScenarioSpec
    ) -> Dict[str, Any]:
        """Decide perturbation using predefined templates and LLM reasoning"""

        # Get available templates for target app
        available_templates = self.templates.get_templates_for_app(execution_context.target_app)

        if not available_templates:
            return self._create_fallback_decision(execution_context.target_app)

        # Use LLM to select appropriate template and parameters
        llm_decision = self._get_llm_template_selection(execution_context, scenario_spec, available_templates)

        if not llm_decision:
            return self._create_fallback_decision(execution_context.target_app)

        # Resolve template parameters
        template = self.templates.get_template_by_name(
            execution_context.target_app, llm_decision["template_name"]
        )

        if not template:
            return self._create_fallback_decision(execution_context.target_app)

        # Start with template defaults and randomized parameters from scenario
        resolved_params = template.parameters.copy()

        # Use randomized parameters from scenario if available (for diversity)
        if hasattr(scenario_spec, "randomized_parameters") and scenario_spec.randomized_parameters:
            resolved_params.update(scenario_spec.randomized_parameters)

        # Override with concrete parameters from LLM if provided
        concrete_params = llm_decision.get("parameters", {})
        resolved_params.update(concrete_params)

        # Generate final command
        generated_command = self._generate_command_from_template(template, resolved_params)

        # Convert resolved_params to PerturbationParameters
        params = PerturbationParameters(
            # Chrome/DOM parameters
            selector=resolved_params.get("selector", ""),
            color=resolved_params.get("color", ""),
            text=resolved_params.get("text", ""),
            radius=resolved_params.get("radius", ""),
            shadow=resolved_params.get("shadow", ""),
            container=resolved_params.get("container", ""),
            direction=resolved_params.get("direction", ""),
            justify=resolved_params.get("justify", ""),
            align=resolved_params.get("align", ""),
            gap=resolved_params.get("gap", ""),
            # System parameters
            theme=resolved_params.get("theme", ""),
            font=resolved_params.get("font", ""),
            title=resolved_params.get("title", ""),
            message=resolved_params.get("message", ""),
            # File system parameters
            old_name=resolved_params.get("old_name", ""),
            new_name=resolved_params.get("new_name", ""),
            folder=resolved_params.get("folder", ""),
            file=resolved_params.get("file", ""),
            # LibreOffice parameters
            col=resolved_params.get("col", ""),
            row=resolved_params.get("row", ""),
            formula=resolved_params.get("formula", ""),
            weight=resolved_params.get("weight", ""),
            # Window management parameters
            x=resolved_params.get("x", ""),
            y=resolved_params.get("y", ""),
            width=resolved_params.get("width", ""),
            height=resolved_params.get("height", ""),
            window_name=resolved_params.get("window_name", ""),
        )

        decision = PerturbationDecision(
            should_apply=llm_decision["should_apply"],
            reasoning=llm_decision["reasoning"],
            template_name=template.name,
            api_call=template.api_call.value,
            generated_command=generated_command,
            parameters=params,
            confidence=llm_decision["confidence"],
            alternative_commands=llm_decision.get("alternative_commands", []),
            visual_impact=llm_decision.get("visual_impact", ""),
            coherence_rationale=llm_decision.get("coherence_rationale", ""),
        )

        # Convert to dictionary format for compatibility with existing code
        return self._convert_decision_to_dict(decision, template)

    def _get_llm_template_selection(
        self,
        execution_context: ExecutionContext,
        scenario_spec: ScenarioSpec,
        available_templates: List[PerturbationTemplate],
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to select appropriate template and parameters"""

        # Format available templates for LLM
        templates_info = self._format_templates_for_llm(available_templates)

        # Format window states for concrete parameter generation
        window_states_info = self._format_window_states_for_perturbation(execution_context.window_states)

        prompt = f"""
You are a perturbation decision system. Select the most appropriate perturbation template and generate concrete parameters based on the execution context and current window states.

EXECUTION CONTEXT:
- Target App: {execution_context.target_app}
- Current Step: {execution_context.step_idx}
- Task Instruction: {execution_context.task_instruction}

SCENARIO REQUIREMENTS:
- Perturbation Type: {scenario_spec.perturbation_types}
- Perturbation Category: {scenario_spec.perturbation_category}
- Intensity: {scenario_spec.perturbation_intensity}
- Learning Objectives: {scenario_spec.learning_objectives}

CURRENT WINDOW STATES:
{window_states_info}

AVAILABLE TEMPLATES:
{templates_info}

DECISION CRITERIA:
1. Choose template that matches scenario requirements
2. Generate CONCRETE parameters based on current window states
3. Consider educational value and learning objectives
4. Ensure safety constraints are met
5. Select appropriate intensity level
6. Maintain coherence with task context

PARAMETER GENERATION GUIDELINES:
- For Chrome: Specify exact selectors (button, input, div), colors (#ff0000), text content, layout changes
- For LibreOffice Calc: Specify exact cells (A1, B2), sheet names, formatting attributes, data values
- For OS/System: Specify exact file names, folder paths, theme names, icon themes
- For VSCode: Specify exact file names, workspace settings, theme names, font families
- Always use realistic, specific values that exist in the current context
- If specific elements are not available, use generic selectors (e.g., "button" instead of "button#submit")
- Prefer generic but safe parameters over specific but potentially invalid ones

Provide your decision as a JSON object with the following structure:
{{
    "should_apply": true/false,
    "reasoning": "explanation of your decision",
    "template_name": "name_of_selected_template",
    "api_call": "type_of_api_call",
    "generated_command": "actual_command_to_execute",
    "parameters": {{
        "selector": "CSS selector like button, h1, .menu",
        "color": "hex color like #ff0000",
        "text": "text content to modify",
        "radius": "border radius like 12px",
        "shadow": "box shadow like 0 4px 8px rgba(0,0,0,0.2)",
        "container": "container selector like .menu, #main-content",
        "direction": "flex direction like column-reverse",
        "justify": "justify content like center, space-between",
        "align": "align items like center, flex-start",
        "gap": "gap between elements like 10px, 1rem",
        "theme": "theme name like Adwaita-dark",
        "font": "font name like Ubuntu 12",
        "title": "notification title",
        "message": "notification message",
        "old_name": "original filename",
        "new_name": "new filename",
        "folder": "folder path",
        "file": "file to create",
        "col": "column index like 0",
        "row": "row index like 0",
        "formula": "cell formula like =1+1",
        "weight": "font weight like 150",
        "x": "x position like 100",
        "y": "y position like 100",
        "width": "window width like 1200",
        "height": "window height like 800",
        "window_name": "window name like Code"
    }},
    "confidence": 0.8,
    "intensity": "low/medium/high",
    "alternative_commands": ["alternative1", "alternative2"],
    "visual_impact": "description of visual changes",
    "coherence_rationale": "why this fits the context"
}}
"""

        try:
            response = self.call_llm(prompt, response_schema=PerturbationDecision)
            return response.model_dump()
        except Exception as e:
            self.logger.error(f"Error in LLM template selection: {e}")
            return None

    def _format_templates_for_llm(self, templates: List[PerturbationTemplate]) -> str:
        """Format templates information for LLM prompt"""
        formatted_templates = []

        for template in templates:
            template_info = f"""
Template: {template.name}
- Category: {template.category.value}
- Description: {template.description}
- Educational Value: {template.educational_value or "General learning"}
- Safety Constraints: {", ".join(template.safety_constraints) if template.safety_constraints else "Standard safety"}
- Target Elements: {", ".join(template.target_elements) if template.target_elements else "Generic elements"}
- Risk Level: {template.risk_level}
"""
            formatted_templates.append(template_info)

        return "\n".join(formatted_templates)

    def _format_window_states_for_perturbation(self, window_states: List[Any]) -> str:
        """Format window states with concrete information for perturbation parameter generation"""
        if not window_states:
            return "No window states available"

        formatted_states = []

        for window_state in window_states:
            app_name = window_state.app_name
            window_name = window_state.window_name

            state_info = f"App: {app_name} - Window: {window_name}\n"

            # Extract concrete elements and information based on app type
            if app_name.lower() in ["chrome", "chromium", "google-chrome"]:
                state_info += self._extract_chrome_elements(window_state)
            elif app_name.lower() in ["code", "vscode"]:
                state_info += self._extract_vscode_elements(window_state)
            elif app_name.lower() in ["libreoffice_calc", "calc"]:
                state_info += self._extract_calc_elements(window_state)
            elif app_name.lower() in ["libreoffice_writer", "writer"]:
                state_info += self._extract_writer_elements(window_state)
            elif app_name.lower() in ["libreoffice_impress", "impress"]:
                state_info += self._extract_impress_elements(window_state)
            else:
                state_info += self._extract_generic_elements(window_state)

            formatted_states.append(state_info)

        return "\n".join(formatted_states)

    def _extract_chrome_elements(self, window_state) -> str:
        """Extract Chrome-specific elements for perturbation"""
        elements_info = []

        if hasattr(window_state, "root_element") and window_state.root_element:
            elements_info.append("Available UI Elements:")

            # Extract buttons, inputs, links, etc.
            buttons = self._find_elements_by_type(window_state.root_element, ["button", "input", "a", "div"])
            if buttons:
                elements_info.append(f"  Buttons/Inputs: {len(buttons)} found")
                for i, btn in enumerate(buttons[:3]):  # Show first 3
                    name = btn.name or f"element_{i}"
                    elements_info.append(f"    - {btn.element_type}: '{name}'")

            # Extract text elements
            text_elements = self._find_elements_by_type(
                window_state.root_element, ["h1", "h2", "h3", "p", "span"]
            )
            if text_elements:
                elements_info.append(f"  Text Elements: {len(text_elements)} found")
                for i, text in enumerate(text_elements[:3]):  # Show first 3
                    name = text.name or f"text_{i}"
                    elements_info.append(f"    - {text.element_type}: '{name}'")

            # Extract containers for layout changes
            containers = self._find_elements_by_type(
                window_state.root_element, ["div", "section", "nav", "header", "footer"]
            )
            if containers:
                elements_info.append(f"  Containers: {len(containers)} found")
                for i, container in enumerate(containers[:3]):  # Show first 3
                    name = container.name or f"container_{i}"
                    elements_info.append(f"    - {container.element_type}: '{name}'")

        return "\n".join(elements_info) if elements_info else "No specific elements found"

    def _extract_vscode_elements(self, window_state) -> str:
        """Extract VSCode-specific elements for perturbation"""
        elements_info = []

        elements_info.append("VSCode Environment:")
        elements_info.append("  - Editor themes available: Dark+, Light+, Monokai, etc.")
        elements_info.append("  - Font families: 'Consolas', 'Monaco', 'Courier New'")
        elements_info.append("  - Window layout: Sidebar, Editor, Terminal panels")

        # Extract file information if available
        if hasattr(window_state, "root_element") and window_state.root_element:
            files = self._find_elements_by_type(window_state.root_element, ["file", "folder"])
            if files:
                elements_info.append(f"  - Files/Folders: {len(files)} found")
                for i, file in enumerate(files[:3]):  # Show first 3
                    name = file.name or f"file_{i}"
                    elements_info.append(f"    - {file.element_type}: '{name}'")

        return "\n".join(elements_info)

    def _extract_calc_elements(self, window_state) -> str:
        """Extract LibreOffice Calc-specific elements for perturbation"""
        elements_info = []

        elements_info.append("LibreOffice Calc Environment:")
        elements_info.append("  - Sheet names: 'Sheet1', 'Sheet2', 'Sheet3'")
        elements_info.append("  - Cell ranges: A1:Z100, B2:D10, etc.")
        elements_info.append("  - Formatting: Font colors, background colors, borders")
        elements_info.append("  - Data types: Numbers, text, formulas")

        # Extract sheet and cell information if available
        if hasattr(window_state, "root_element") and window_state.root_element:
            cells = self._find_elements_by_type(window_state.root_element, ["cell", "sheet"])
            if cells:
                elements_info.append(f"  - Cells/Sheets: {len(cells)} found")
                for i, cell in enumerate(cells[:3]):  # Show first 3
                    name = cell.name or f"cell_{i}"
                    elements_info.append(f"    - {cell.element_type}: '{name}'")

        return "\n".join(elements_info)

    def _extract_writer_elements(self, window_state) -> str:
        """Extract LibreOffice Writer-specific elements for perturbation"""
        elements_info = []

        elements_info.append("LibreOffice Writer Environment:")
        elements_info.append("  - Text formatting: Font families, sizes, colors")
        elements_info.append("  - Paragraph styles: Heading 1, Heading 2, Normal")
        elements_info.append("  - Document elements: Headers, footers, tables")

        # Extract text elements if available
        if hasattr(window_state, "root_element") and window_state.root_element:
            text_elements = self._find_elements_by_type(window_state.root_element, ["text", "paragraph"])
            if text_elements:
                elements_info.append(f"  - Text Elements: {len(text_elements)} found")
                for i, text in enumerate(text_elements[:3]):  # Show first 3
                    name = text.name or f"text_{i}"
                    elements_info.append(f"    - {text.element_type}: '{name}'")

        return "\n".join(elements_info)

    def _extract_impress_elements(self, window_state) -> str:
        """Extract LibreOffice Impress-specific elements for perturbation"""
        elements_info = []

        elements_info.append("LibreOffice Impress Environment:")
        elements_info.append("  - Slide backgrounds: Colors, gradients, images")
        elements_info.append("  - Text boxes: Titles, content, captions")
        elements_info.append("  - Slide layouts: Title slide, content slide, etc.")

        # Extract slide elements if available
        if hasattr(window_state, "root_element") and window_state.root_element:
            slides = self._find_elements_by_type(window_state.root_element, ["slide", "textbox"])
            if slides:
                elements_info.append(f"  - Slides/Textboxes: {len(slides)} found")
                for i, slide in enumerate(slides[:3]):  # Show first 3
                    name = slide.name or f"slide_{i}"
                    elements_info.append(f"    - {slide.element_type}: '{name}'")

        return "\n".join(elements_info)

    def _extract_generic_elements(self, window_state) -> str:
        """Extract generic elements for system-level perturbations"""
        elements_info = []

        elements_info.append("System Environment:")
        elements_info.append("  - Available themes: Adwaita, Adwaita-dark, HighContrast")
        elements_info.append("  - Font families: Ubuntu, Liberation Sans, DejaVu Sans")
        elements_info.append("  - Icon themes: Papirus, Papirus-Dark, Adwaita")
        elements_info.append("  - File system: /tmp/, /home/, current directory")

        # Extract file elements if available
        if hasattr(window_state, "root_element") and window_state.root_element:
            files = self._find_elements_by_type(window_state.root_element, ["file", "folder"])
            if files:
                elements_info.append(f"  - Files/Folders: {len(files)} found")
                for i, file in enumerate(files[:3]):  # Show first 3
                    name = file.name or f"file_{i}"
                    elements_info.append(f"    - {file.element_type}: '{name}'")

        return "\n".join(elements_info)

    def _find_elements_by_type(self, root_element, element_types: List[str]) -> List[Any]:
        """Find elements by type in the element tree"""
        found_elements = []

        def search_element(element):
            if hasattr(element, "element_type") and element.element_type in element_types:
                found_elements.append(element)

            if hasattr(element, "children"):
                for child in element.children:
                    search_element(child)

        search_element(root_element)
        return found_elements

    def _generate_command_from_template(
        self, template: PerturbationTemplate, resolved_params: Dict[str, Any]
    ) -> str:
        """Generate final command from template and resolved parameters"""
        try:
            return template.template_command.format(**resolved_params)
        except KeyError as e:
            self.logger.error(f"Missing parameter {e} for template {template.name}")
            return template.template_command

    def _create_fallback_decision(self, target_app: str) -> Dict[str, Any]:
        """Create safe fallback decision when template selection fails"""
        fallback_decision = PerturbationDecision(
            should_apply=False,
            reasoning="Template selection failed - using safe fallback",
            template_name="fallback",
            api_call="execute_bash_command",
            generated_command=f'echo "Safe fallback for {target_app}"',
            parameters=PerturbationParameters(),
            confidence=0.1,
            alternative_commands=[],
            visual_impact="No visual changes - safe fallback",
            coherence_rationale="Fallback response to prevent errors",
        )

        # Convert to dictionary format for compatibility
        return self._convert_decision_to_dict(fallback_decision, None)

    def _convert_decision_to_dict(
        self, decision: PerturbationDecision, template: Optional[PerturbationTemplate]
    ) -> Dict[str, Any]:
        """Convert PerturbationDecision to dictionary format for compatibility with existing code"""
        return {
            "should_apply": decision.should_apply,
            "reasoning": decision.reasoning,
            "perturbation_type": template.category.value if template else "unknown",
            "target_app": "unknown",  # Will be set by caller if needed
            "api_call": decision.api_call,
            "generated_command": decision.generated_command,
            "parameters": decision.parameters.model_dump(),
            "confidence": decision.confidence,
            "alternative_commands": decision.alternative_commands,
            "visual_impact": decision.visual_impact,
            "coherence_rationale": decision.coherence_rationale,
            "template_name": decision.template_name,
        }


class ParameterRandomizer:
    """Simple parameter randomization for realistic distribution"""

    def __init__(self):
        # Real-world parameter distributions (simplified)
        self.color_palette = [
            "#ff0000",
            "#00ff00",
            "#0000ff",
            "#ffff00",
            "#ff00ff",
            "#00ffff",
            "#ff6b6b",
            "#4ecdc4",
            "#45b7d1",
            "#96ceb4",
            "#feca57",
            "#ff9ff3",
            "#54a0ff",
            "#5f27cd",
            "#00d2d3",
            "#ff9f43",
            "#ee5a24",
            "#c44569",
        ]

        self.theme_options = [
            "Adwaita",
            "Adwaita-dark",
            "HighContrast",
            "HighContrastInverse",
            "Yaru",
            "Yaru-dark",
            "Arc",
            "Arc-dark",
            "Numix",
            "Numix-dark",
        ]

        self.font_options = [
            "Ubuntu 12",
            "Liberation Sans 14",
            "DejaVu Sans 12",
            "Noto Sans 13",
            "Source Sans Pro 12",
            "Roboto 14",
            "Open Sans 12",
            "Lato 13",
        ]

        self.text_variations = [
            "Modified Text",
            "Updated Content",
            "New Label",
            "Changed Text",
            "Altered Content",
            "Revised Text",
            "Updated Label",
            "Modified Content",
        ]

        self.selector_variations = {
            "button": ["button", "input[type='button']", ".btn", "button[class*='button']"],
            "text": ["h1", "h2", "h3", "p", "span", ".text", ".label"],
            "container": [".menu", ".nav", ".header", ".content", ".main", ".container"],
        }

    def randomize_parameters(self, template: PerturbationTemplate) -> Dict[str, Any]:
        """Generate randomized parameters for a template - DRAMATIC VALUES"""
        import random

        params = template.parameters.copy()

        # Randomize based on template type with DRAMATIC values
        if "dramatic_color_inversion" in template.name:
            params["bg_color"] = random.choice(["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"])

        elif "massive_font_change" in template.name:
            params["size"] = random.choice(["28px", "32px", "36px", "40px"])
            params["font"] = random.choice(["Impact", "Arial Black", "Comic Sans MS", "Papyrus"])

        elif "rainbow_background" in template.name:
            # No parameters needed for rainbow background
            pass

        elif "element_rotation" in template.name:
            params["selector"] = random.choice(["button", "input", "div", "h1", "h2", "p"])
            params["angle"] = str(random.choice([15, 30, 45, -15, -30, -45]))

        elif "blinking_elements" in template.name:
            params["selector"] = random.choice(["button", "a", "h1", "h2", "input", "div"])

        elif "dramatic_theme_change" in template.name:
            params["theme"] = random.choice(["HighContrast", "HighContrastInverse", "Adwaita-dark"])

        elif "massive_font_change" in template.name or "massive_monospace_font" in template.name:
            params["font"] = random.choice(["Liberation Sans 20", "Ubuntu 22", "DejaVu Sans 24"])
            params["mono_font"] = random.choice(["Liberation Mono 22", "Ubuntu Mono 24", "Courier New 20"])

        elif "dramatic_icon_change" in template.name:
            params["theme"] = random.choice(["HighContrast", "HighContrastInverse", "Adwaita"])

        elif (
            "persistent_notification_spam" in template.name
            or "persistent_calc_notifications" in template.name
            or "persistent_dev_notifications" in template.name
        ):
            params["title"] = random.choice(
                ["URGENT ALERT", "CRITICAL ERROR", "SYSTEM FAILURE", "IMMEDIATE ATTENTION"]
            )
            params["message"] = random.choice(
                [
                    "Multiple critical errors detected!",
                    "System instability detected!",
                    "Immediate action required!",
                    "Critical failure imminent!",
                ]
            )

        elif "dramatic_cell_colors" in template.name:
            params["color"] = random.choice(["0xFF0000", "0x00FF00", "0x0000FF", "0xFFFF00", "0xFF00FF"])

        elif "sheet_name_change" in template.name:
            params["new_name"] = random.choice(
                ["CONFUSING_SHEET_123", "RANDOM_DATA_456", "UNKNOWN_SHEET_789", "MYSTERY_SHEET_ABC"]
            )

        elif "confusing_file_names" in template.name:
            params["old_name"] = random.choice(["main.py", "app.js", "index.html", "config.json"])
            params["new_name"] = random.choice(
                [
                    "CONFUSING_MAIN_FILE_123.py",
                    "RANDOM_APP_FILE_456.js",
                    "MYSTERY_INDEX_789.html",
                    "UNKNOWN_CONFIG_ABC.json",
                ]
            )

        elif "workspace_clutter" in template.name:
            params["folder"] = random.choice(["src", "lib", "test", "docs"])

        elif "window_repositioning" in template.name:
            params["x"] = random.choice([10, 50, 100, 200])
            params["y"] = random.choice([10, 50, 100, 200])

        return params


class CurriculumGenerator(BaseLLM):
    """Simplified curriculum generator focused on scenario generation"""

    def __init__(self, model_name: str = "gemini-2.0-flash-lite", model_provider: str = "gemini"):
        super().__init__(model_name, model_provider)
        self.templates = AppPerturbationTemplates()
        self.parameter_randomizer = ParameterRandomizer()
        self.logger = logging.getLogger(__name__)

    def generate_scenario_specs(
        self, seed_trajectory, window_states: List[Any], curriculum_config
    ) -> List[ScenarioSpec]:
        """Generate scenario specifications using simplified approach"""

        # Analyze task context
        task_context = self._analyze_task_context(seed_trajectory, window_states)

        # Generate scenarios based on available templates
        scenarios = self._generate_scenarios_from_templates(task_context, curriculum_config.scenario_count)

        return scenarios

    def _analyze_task_context(self, seed_trajectory, window_states: List[Any]) -> Dict[str, Any]:
        """Analyze task context to understand requirements"""
        return {
            "task_id": seed_trajectory.task_id,
            "task_instruction": seed_trajectory.task_instruction,
            "target_apps": [ws.app_name for ws in window_states],
            "available_templates": {
                app: len(self.templates.get_templates_for_app(app))
                for app in [ws.app_name for ws in window_states]
            },
        }

    def _generate_scenarios_from_templates(
        self, task_context: Dict[str, Any], target_count: int
    ) -> List[ScenarioSpec]:
        """Generate scenarios based on available templates with random distribution"""
        import random

        scenarios = []
        target_apps = task_context["target_apps"]

        if not target_apps:
            return scenarios

        # Calculate scenarios per app with better distribution
        scenarios_per_app = max(1, target_count // len(target_apps))
        remaining_scenarios = target_count % len(target_apps)

        for i, app_name in enumerate(target_apps):
            templates = self.templates.get_templates_for_app(app_name)

            if not templates:
                continue

            # Give extra scenarios to first few apps if there's a remainder
            app_scenario_count = scenarios_per_app
            if i < remaining_scenarios:
                app_scenario_count += 1

            # Random template selection for diversity
            if len(templates) >= app_scenario_count:
                templates_to_use = random.sample(templates, app_scenario_count)
            else:
                # If we need more scenarios than templates, repeat randomly
                templates_to_use = []
                for _ in range(app_scenario_count):
                    templates_to_use.append(random.choice(templates))

            for template in templates_to_use:
                # Generate randomized parameters for diversity
                randomized_params = self.parameter_randomizer.randomize_parameters(template)

                scenario = ScenarioSpec(
                    scenario_id=f"{task_context['task_id']}_scenario_{len(scenarios) + 1}_{app_name}",
                    scenario_index=len(scenarios),
                    target_app=app_name,
                    perturbation_trigger="template_based",
                    available_perturbation_actions=template.name,
                    learning_objectives=template.educational_value or "General learning objectives",
                    target_components=template.target_elements or ["generic"],
                    perturbation_types=[PerturbationType(template.category.value)],
                    perturbation_category=self._map_template_category_to_scenario_category(template.category),
                    perturbation_intensity=PerturbationIntensity.from_string(
                        "medium", PerturbationIntensity.MEDIUM
                    ),
                    maintains_functionality=True,
                    maintains_accessibility=True,
                    realistic_scenario=f"Apply {template.name} to {app_name}",
                    initial_state_perturbation=False,
                    runtime_perturbation=True,
                    risk_mitigation="Built-in safety constraints",
                    educational_rationale=template.educational_value or "General educational value",
                )

                # Store randomized parameters for later use in perturbation generation
                scenario.randomized_parameters = randomized_params
                scenarios.append(scenario)

                # Stop if we've reached the target count
                if len(scenarios) >= target_count:
                    break

            # Stop if we've reached the target count
            if len(scenarios) >= target_count:
                break

        return scenarios[:target_count]

    def _map_template_category_to_scenario_category(
        self, template_category: TemplateCategory
    ) -> PerturbationCategory:
        """Map template category to ScenarioSpec category"""
        # Map template categories to scenario categories
        mapping = {
            TemplateCategory.VISUAL: PerturbationCategory.CONTENT_RANDOMIZATION,
            TemplateCategory.SYSTEM: PerturbationCategory.SYSTEM_LEVEL,
            TemplateCategory.CONTENT: PerturbationCategory.CONTENT_RANDOMIZATION,
            TemplateCategory.LAYOUT: PerturbationCategory.CONTENT_RANDOMIZATION,
            TemplateCategory.THEME: PerturbationCategory.SYSTEM_LEVEL,
            TemplateCategory.NOTIFICATION: PerturbationCategory.CROSS_APP_INTERFERENCE,
            TemplateCategory.FILE_SYSTEM: PerturbationCategory.SYSTEM_LEVEL,
            TemplateCategory.WINDOW_MANAGEMENT: PerturbationCategory.SYSTEM_LEVEL,
        }

        return mapping.get(template_category, PerturbationCategory.SYSTEM_LEVEL)
