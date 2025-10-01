"""
Comprehensive App State Extractor

Extracts rich UI state information for LLM consumption using:
- Browser: Playwright/CDP for DOM tree extraction
- LibreOffice: Accessibility tree + UNO API
- Other apps: Enhanced accessibility tree parsing

Inspired by extract_ui_coordinates.py but focused on extracting
comprehensive state rather than just coordinates.
"""

import logging
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppStateExtractor:
    """Extract comprehensive app state information for LLM prompting."""

    def __init__(self, controller):
        """
        Args:
            controller: PerturbationController with access to:
                - get_accessibility_tree()
                - _get_page() for Playwright
                - execute_python_command()
        """
        self.controller = controller
        self.logger = logging.getLogger(__name__)

    def extract_app_states(self, use_comprehensive: bool = True) -> List[Dict[str, Any]]:
        """
        Extract app states for all visible applications.

        Args:
            use_comprehensive: If True, use comprehensive extraction (DOM, UNO, etc.)
                             If False, use basic accessibility tree only

        Returns list of app states, each containing:
        - app_type: browser, libreoffice_calc, etc.
        - app_name: Full application name
        - current_view: Detected view type
        - interactive_elements: Buttons, links, inputs with details (comprehensive only)
        - content_structure: Hierarchy of main content areas (comprehensive only)
        - active_dialogs: Any open dialogs/modals
        - menu_structure: Available menus and items
        - metadata: Additional app-specific information
        """

        # Get accessibility tree
        a11y_tree = self.controller.get_accessibility_tree()
        if not a11y_tree:
            self.logger.warning("No accessibility tree available")
            return []

        try:
            root = ET.fromstring(a11y_tree)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse accessibility tree: {e}")
            return []

        # Group elements by application
        app_groups = self._group_elements_by_app(root)
        app_states = []

        for app_name, elements in app_groups.items():
            app_type = self._detect_app_type(app_name)

            if app_type == "unknown":
                continue

            # Extract state based on mode and app type
            if use_comprehensive:
                # Comprehensive extraction with app-specific enhancements
                if app_type == "browser":
                    app_state = self._extract_browser_state(app_name, elements)
                elif app_type in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
                    app_state = self._extract_libreoffice_state(app_name, app_type, elements)
                else:
                    app_state = self._extract_generic_state(app_name, app_type, elements)
            else:
                # Basic extraction (lightweight, fast)
                app_state = self._extract_basic_state(app_name, app_type, elements)

            app_states.append(app_state)

        # If no apps found, return placeholder
        if not app_states:
            app_states.append(
                {
                    "app_type": "unknown",
                    "app_name": "unknown",
                    "current_view": "unknown",
                    "key_elements": [],
                    "task_context": "No accessible applications detected",
                    "element_count": 0,
                }
            )

        return app_states

    # ==================== BROWSER STATE EXTRACTION ====================

    def _extract_browser_state(self, app_name: str, elements: List[ET.Element]) -> Dict[str, Any]:
        """
        Extract comprehensive browser state using Playwright/CDP.

        Returns rich DOM information that LLMs can understand.
        """
        self.logger.info(f"Extracting browser state for {app_name}")

        # Base state from accessibility tree
        base_state = self._extract_generic_state(app_name, "browser", elements)

        # Try to get richer DOM information via Playwright
        try:
            page = self.controller._get_page()
            if page:
                # Extract DOM structure
                dom_state = self._extract_dom_structure(page)
                base_state.update(dom_state)

        except Exception as e:
            self.logger.warning(f"Could not extract DOM state via Playwright: {e}")

        return base_state

    def _extract_dom_structure(self, page) -> Dict[str, Any]:
        """
        Extract comprehensive DOM structure using CDP.

        Similar to icon extraction but captures ALL interactive elements.
        """
        try:
            js_code = """
            () => {
                const result = {
                    url: window.location.href,
                    title: document.title,
                    interactive_elements: [],
                    forms: [],
                    links: [],
                    buttons: [],
                    inputs: [],
                    headings: [],
                    images: [],
                    content_sections: []
                };

                // Helper: Check visibility
                function isVisible(element) {
                    const style = window.getComputedStyle(element);
                    const rect = element.getBoundingClientRect();
                    return (
                        style.display !== 'none' &&
                        style.visibility !== 'hidden' &&
                        style.opacity !== '0' &&
                        rect.width > 0 && rect.height > 0
                    );
                }

                // Helper: Get element descriptor
                function getElementDescriptor(element) {
                    return {
                        tag: element.tagName.toLowerCase(),
                        text: (element.innerText || element.textContent || '').trim().substring(0, 100),
                        aria_label: element.getAttribute('aria-label') || '',
                        id: element.id || '',
                        class: element.className || '',
                        name: element.getAttribute('name') || '',
                        type: element.getAttribute('type') || '',
                        placeholder: element.getAttribute('placeholder') || '',
                        value: element.value || '',
                        href: element.href || '',
                        role: element.getAttribute('role') || '',
                        title: element.getAttribute('title') || ''
                    };
                }

                // Extract buttons
                const buttons = document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]');
                buttons.forEach(btn => {
                    if (isVisible(btn)) {
                        result.buttons.push(getElementDescriptor(btn));
                    }
                });

                // Extract links
                const links = document.querySelectorAll('a[href]');
                links.forEach(link => {
                    if (isVisible(link)) {
                        result.links.push(getElementDescriptor(link));
                    }
                });

                // Extract input fields
                const inputs = document.querySelectorAll('input:not([type="hidden"]), textarea, select');
                inputs.forEach(input => {
                    if (isVisible(input)) {
                        result.inputs.push(getElementDescriptor(input));
                    }
                });

                // Extract forms
                const forms = document.querySelectorAll('form');
                forms.forEach(form => {
                    if (isVisible(form)) {
                        const formData = getElementDescriptor(form);
                        formData.fields = Array.from(form.querySelectorAll('input, textarea, select'))
                            .map(f => ({
                                type: f.type,
                                name: f.name,
                                placeholder: f.placeholder,
                                required: f.required
                            }));
                        result.forms.push(formData);
                    }
                });

                // Extract headings (content structure)
                const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
                headings.forEach(heading => {
                    if (isVisible(heading)) {
                        result.headings.push({
                            level: heading.tagName.toLowerCase(),
                            text: heading.textContent.trim().substring(0, 100)
                        });
                    }
                });

                // Extract images with alt text
                const images = document.querySelectorAll('img[alt]');
                images.forEach(img => {
                    if (isVisible(img)) {
                        result.images.push({
                            alt: img.alt,
                            src: img.src.substring(0, 100),
                            title: img.title
                        });
                    }
                });

                // Extract main content sections
                const sections = document.querySelectorAll('main, article, section, nav, aside, header, footer');
                sections.forEach(section => {
                    if (isVisible(section)) {
                        result.content_sections.push({
                            tag: section.tagName.toLowerCase(),
                            role: section.getAttribute('role') || '',
                            aria_label: section.getAttribute('aria-label') || '',
                            text_preview: section.textContent.trim().substring(0, 150)
                        });
                    }
                });

                // Combine all interactive elements
                result.interactive_elements = [
                    ...result.buttons.map(b => ({...b, element_type: 'button'})),
                    ...result.links.map(l => ({...l, element_type: 'link'})),
                    ...result.inputs.map(i => ({...i, element_type: 'input'}))
                ];

                // Limit arrays to prevent overwhelming LLM
                result.buttons = result.buttons.slice(0, 20);
                result.links = result.links.slice(0, 30);
                result.inputs = result.inputs.slice(0, 20);
                result.headings = result.headings.slice(0, 15);
                result.images = result.images.slice(0, 10);
                result.content_sections = result.content_sections.slice(0, 10);
                result.interactive_elements = result.interactive_elements.slice(0, 50);

                return result;
            }
            """

            dom_data = page.evaluate(js_code)

            self.logger.info(
                f"Extracted DOM: {len(dom_data['buttons'])} buttons, "
                f"{len(dom_data['links'])} links, {len(dom_data['inputs'])} inputs"
            )

            return {
                "dom_extracted": True,
                "page_url": dom_data["url"],
                "page_title": dom_data["title"],
                "buttons": dom_data["buttons"],
                "links": dom_data["links"],
                "input_fields": dom_data["inputs"],
                "forms": dom_data["forms"],
                "headings": dom_data["headings"],
                "images": dom_data["images"],
                "content_sections": dom_data["content_sections"],
                "interactive_elements_summary": {
                    "total_buttons": len(dom_data["buttons"]),
                    "total_links": len(dom_data["links"]),
                    "total_inputs": len(dom_data["inputs"]),
                    "total_forms": len(dom_data["forms"]),
                },
            }

        except Exception as e:
            self.logger.error(f"Error extracting DOM structure: {e}")
            return {"dom_extracted": False, "error": str(e)}

    # ==================== LIBREOFFICE STATE EXTRACTION ====================

    def _extract_libreoffice_state(
        self, app_name: str, app_type: str, elements: List[ET.Element]
    ) -> Dict[str, Any]:
        """
        Extract LibreOffice state using accessibility tree + UNO API.
        """
        self.logger.info(f"Extracting LibreOffice state for {app_name}")

        # Base state from accessibility tree
        base_state = self._extract_generic_state(app_name, app_type, elements)

        # Try to get document-specific information via UNO
        try:
            uno_state = self._extract_libreoffice_uno_state(app_type)
            if uno_state:
                base_state.update(uno_state)
        except Exception as e:
            self.logger.warning(f"Could not extract UNO state: {e}")

        return base_state

    def _extract_libreoffice_uno_state(self, app_type: str) -> Optional[Dict[str, Any]]:
        """
        Extract document state using UNO API.
        """
        try:
            if app_type == "libreoffice_calc":
                uno_code = """
# Extract Calc-specific state
doc = desktop.getCurrentComponent()
if doc:
    sheets = doc.getSheets()
    sheet_names = [sheets.getByIndex(i).getName() for i in range(min(sheets.getCount(), 10))]

    active_sheet = doc.getCurrentController().getActiveSheet()
    active_sheet_name = active_sheet.getName()

    # Get some cell values for context
    sample_cells = []
    for row in range(min(5, active_sheet.getRows().getCount())):
        for col in range(min(5, active_sheet.getColumns().getCount())):
            cell = active_sheet.getCellByPosition(col, row)
            value = cell.getString() if cell.getType().value == 'TEXT' else str(cell.getValue())
            if value:
                sample_cells.append(f"({col},{row}): {value[:50]}")

    print(f"SHEETS: {','.join(sheet_names)}")
    print(f"ACTIVE_SHEET: {active_sheet_name}")
    print(f"SAMPLE_CELLS: {' | '.join(sample_cells[:20])}")
"""
            elif app_type == "libreoffice_writer":
                uno_code = """
# Extract Writer-specific state
doc = desktop.getCurrentComponent()
if doc:
    text = doc.getText()
    cursor = text.createTextCursor()
    cursor.gotoStart(False)
    cursor.gotoEnd(True)

    content_preview = text.getString()[:500]

    # Get paragraph count
    paragraphs = doc.getTextContent()
    para_count = paragraphs.getCount() if hasattr(paragraphs, 'getCount') else 0

    print(f"CONTENT_PREVIEW: {content_preview}")
    print(f"PARAGRAPH_COUNT: {para_count}")
"""
            elif app_type == "libreoffice_impress":
                uno_code = """
# Extract Impress-specific state
doc = desktop.getCurrentComponent()
if doc:
    slides = doc.getDrawPages()
    slide_count = slides.getCount()

    current_controller = doc.getCurrentController()
    current_slide_index = current_controller.getCurrentPage().getImplementationName()

    slide_titles = []
    for i in range(min(slide_count, 10)):
        slide = slides.getByIndex(i)
        # Try to get slide title
        if hasattr(slide, 'getName'):
            slide_titles.append(slide.getName())

    print(f"SLIDE_COUNT: {slide_count}")
    print(f"SLIDE_TITLES: {','.join(slide_titles)}")
"""
            else:
                return None

            result = self.controller.execute_uno_command(uno_code, {})

            # Parse UNO output
            if result and result.get("status") == "success":
                output = result.get("output", "")
                # Parse the printed output
                # This is simplified - actual parsing would be more robust
                uno_data = {}
                for line in output.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        uno_data[key.strip().lower().replace(" ", "_")] = value.strip()

                return {"uno_extracted": True, "document_state": uno_data}

        except Exception as e:
            self.logger.error(f"Error extracting UNO state: {e}")

        return None

    # ==================== GENERIC STATE EXTRACTION ====================

    def _extract_generic_state(
        self, app_name: str, app_type: str, elements: List[ET.Element]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive state from accessibility tree.

        This works for ALL applications (GTK apps, Electron apps, etc.)
        """
        # Parse accessibility tree elements
        parsed_elements = self._parse_accessibility_elements(elements)

        # Categorize elements
        categorized = self._categorize_elements(parsed_elements)

        # Detect UI structure
        ui_structure = self._detect_ui_structure(parsed_elements)

        # Extract interactive elements
        interactive_elements = self._extract_interactive_elements(parsed_elements)

        # Detect current view/state
        current_view = self._detect_current_view_enhanced(parsed_elements)

        # Detect active dialogs
        active_dialogs = self._detect_active_dialogs(parsed_elements)

        # Extract menu structure
        menu_structure = self._extract_menu_structure(parsed_elements)

        return {
            "app_type": app_type,
            "app_name": app_name,
            "current_view": current_view,
            "element_count": len(parsed_elements),
            # Categorized elements
            "buttons": categorized["buttons"],
            "text_fields": categorized["text_fields"],
            "labels": categorized["labels"],
            "menus": categorized["menus"],
            "menu_items": categorized["menu_items"],
            "checkboxes": categorized["checkboxes"],
            "radio_buttons": categorized["radio_buttons"],
            "combo_boxes": categorized["combo_boxes"],
            "tabs": categorized["tabs"],
            "panels": categorized["panels"],
            "scrollbars": categorized["scrollbars"],
            "tables": categorized["tables"],
            # UI structure
            "ui_structure": ui_structure,
            "interactive_elements": interactive_elements[:50],  # Limit for LLM
            "active_dialogs": active_dialogs,
            "menu_structure": menu_structure,
            # Summary statistics
            "summary": {
                "total_buttons": len(categorized["buttons"]),
                "total_text_fields": len(categorized["text_fields"]),
                "total_menus": len(categorized["menus"]),
                "total_menu_items": len(categorized["menu_items"]),
                "total_interactive": len(interactive_elements),
                "has_active_dialog": len(active_dialogs) > 0,
            },
        }

    def _parse_accessibility_elements(self, elements: List[ET.Element]) -> List[Dict[str, Any]]:
        """
        Parse accessibility tree elements into structured dictionaries.

        Extracts ALL available information from each element.
        """
        parsed = []

        for elem in elements:
            # Extract all attributes
            element_data = {
                "tag": elem.tag,
                "role": elem.get("role", ""),
                "name": elem.get("name", ""),
                "description": elem.get("description", ""),
                "value": elem.get("value", ""),
                "text": elem.text if elem.text else "",
                "attributes": {},
            }

            # Extract namespaced attributes
            for key, value in elem.attrib.items():
                # Parse namespace prefixes
                if "}" in key:
                    namespace, attr_name = key.split("}", 1)
                    namespace = namespace.lstrip("{")

                    # Store with friendly names
                    if "component" in namespace:
                        element_data["attributes"][f"component_{attr_name}"] = value
                    elif "state" in namespace:
                        element_data["attributes"][f"state_{attr_name}"] = value
                    elif "attribute" in namespace:
                        element_data["attributes"][f"attr_{attr_name}"] = value
                    else:
                        element_data["attributes"][attr_name] = value
                else:
                    element_data["attributes"][key] = value

            # Extract coordinates if available
            screencoord = element_data["attributes"].get("component_screencoord", "")
            size = element_data["attributes"].get("component_size", "")

            if screencoord and size:
                try:
                    coords = eval(screencoord)
                    sizes = eval(size)
                    element_data["position"] = {
                        "x": coords[0],
                        "y": coords[1],
                        "width": sizes[0],
                        "height": sizes[1],
                    }
                except Exception as e:
                    self.logger.error(f"Error parsing coordinates: {e}")
                    pass

            # Check visibility
            element_data["visible"] = element_data["attributes"].get("state_visible", "") == "true"
            element_data["enabled"] = element_data["attributes"].get("state_enabled", "") == "true"
            element_data["focused"] = element_data["attributes"].get("state_focused", "") == "true"

            parsed.append(element_data)

        return parsed

    def _categorize_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Categorize elements by type for easy access."""
        categories = defaultdict(list)

        for elem in elements:
            if not elem["visible"]:
                continue

            role = elem["role"].lower()
            tag = elem["tag"].lower()

            # Categorize by role/tag
            if "button" in role or "push-button" in tag:
                categories["buttons"].append(
                    {
                        "name": elem["name"],
                        "text": elem["text"],
                        "description": elem["description"],
                        "enabled": elem["enabled"],
                    }
                )

            elif "text" in role or "entry" in tag or "text-box" in role:
                categories["text_fields"].append(
                    {
                        "name": elem["name"],
                        "value": elem["value"],
                        "placeholder": elem["attributes"].get("attr_placeholder", ""),
                        "enabled": elem["enabled"],
                    }
                )

            elif "label" in role or "label" in tag:
                categories["labels"].append(
                    {"text": elem["name"] or elem["text"], "for": elem["attributes"].get("attr_for", "")}
                )

            elif "menu-bar" in role or "menu-bar" in tag:
                categories["menus"].append(
                    {
                        "name": elem["name"],
                        "items": [],  # Would be populated by child elements
                    }
                )

            elif "menu-item" in role or "menu-item" in tag:
                categories["menu_items"].append(
                    {"name": elem["name"], "description": elem["description"], "enabled": elem["enabled"]}
                )

            elif "check-box" in role:
                categories["checkboxes"].append(
                    {
                        "name": elem["name"],
                        "checked": elem["attributes"].get("state_checked", "") == "true",
                        "enabled": elem["enabled"],
                    }
                )

            elif "radio-button" in role:
                categories["radio_buttons"].append(
                    {
                        "name": elem["name"],
                        "selected": elem["attributes"].get("state_selected", "") == "true",
                        "enabled": elem["enabled"],
                    }
                )

            elif "combo-box" in role or "combo-box" in tag:
                categories["combo_boxes"].append(
                    {"name": elem["name"], "value": elem["value"], "enabled": elem["enabled"]}
                )

            elif "tab" in role:
                categories["tabs"].append(
                    {"name": elem["name"], "selected": elem["attributes"].get("state_selected", "") == "true"}
                )

            elif "panel" in role or "scroll-pane" in tag:
                categories["panels"].append({"name": elem["name"], "description": elem["description"]})

            elif "scroll-bar" in role:
                categories["scrollbars"].append(
                    {
                        "name": elem["name"],
                        "orientation": "vertical"
                        if "vertical" in elem["attributes"].get("attr_orientation", "")
                        else "horizontal",
                    }
                )

            elif "table" in role:
                categories["tables"].append(
                    {
                        "name": elem["name"],
                        "rows": elem["attributes"].get("attr_rows", ""),
                        "columns": elem["attributes"].get("attr_columns", ""),
                    }
                )

        return dict(categories)

    def _extract_interactive_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract ALL interactive elements that LLM might want to manipulate."""
        interactive_roles = [
            "button",
            "push-button",
            "menu-item",
            "text",
            "entry",
            "combo-box",
            "check-box",
            "radio-button",
            "link",
            "tab",
            "toggle-button",
            "tool-bar-button",
        ]

        interactive = []
        for elem in elements:
            if not elem["visible"] or not elem["enabled"]:
                continue

            role = elem["role"].lower()
            if any(r in role for r in interactive_roles):
                interactive.append(
                    {
                        "type": role,
                        "name": elem["name"],
                        "text": elem["text"],
                        "description": elem["description"],
                        "value": elem["value"],
                        "position": elem.get("position", {}),
                        "focused": elem["focused"],
                    }
                )

        return interactive

    def _detect_ui_structure(self, elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect high-level UI structure."""
        structure = {
            "has_menu_bar": False,
            "has_toolbar": False,
            "has_status_bar": False,
            "has_sidebar": False,
            "has_main_content": False,
            "dialog_count": 0,
            "window_count": 0,
        }

        for elem in elements:
            role = elem["role"].lower()
            tag = elem["tag"].lower()

            if "menu-bar" in role:
                structure["has_menu_bar"] = True
            elif "tool-bar" in role:
                structure["has_toolbar"] = True
            elif "status-bar" in role:
                structure["has_status_bar"] = True
            elif "side-bar" in tag or "panel" in role:
                structure["has_sidebar"] = True
            elif "dialog" in role:
                structure["dialog_count"] += 1
            elif "frame" in tag or "window" in tag:
                structure["window_count"] += 1

        return structure

    def _detect_current_view_enhanced(self, elements: List[Dict[str, Any]]) -> str:
        """Enhanced view detection based on visible elements."""
        roles = [elem["role"].lower() for elem in elements if elem["visible"]]

        # Check for specific views
        if any("dialog" in r for r in roles):
            # Identify dialog type
            dialog_names = [
                elem["name"] for elem in elements if "dialog" in elem["role"].lower() and elem["visible"]
            ]
            return f"dialog_view ({', '.join(dialog_names[:2])})"

        elif any("menu" in r for r in roles) and not any("menu-bar" in r for r in roles):
            return "menu_expanded"

        elif sum("text" in r or "entry" in r for r in roles) > 3:
            return "form_view"

        elif any("table" in r for r in roles):
            return "table_view"

        elif sum("tab" in r for r in roles) > 2:
            return "tabbed_view"

        else:
            return "main_view"

    def _detect_active_dialogs(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect any active dialogs with their content."""
        dialogs = []

        for elem in elements:
            if "dialog" in elem["role"].lower() and elem["visible"]:
                dialogs.append(
                    {
                        "name": elem["name"],
                        "description": elem["description"],
                        "modal": elem["attributes"].get("state_modal", "") == "true",
                    }
                )

        return dialogs

    def _extract_menu_structure(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract menu structure for understanding available actions."""
        menus = []

        for elem in elements:
            if "menu-bar" in elem["role"].lower() or "menu" in elem["role"].lower():
                if elem["visible"]:
                    menus.append({"name": elem["name"], "type": elem["role"], "enabled": elem["enabled"]})

        return menus

    # ==================== BASIC STATE EXTRACTION ====================

    def _extract_basic_state(
        self, app_name: str, app_type: str, elements: List[ET.Element]
    ) -> Dict[str, Any]:
        """
        Basic state extraction - lightweight, fast, uses only accessibility tree.

        This is the original implementation from perturbation_desktop_env.py.
        Used as fallback when comprehensive extraction is not needed or fails.
        """
        # Convert ET.Element list to dict list for processing
        element_dicts = []
        for elem in elements:
            element_dicts.append(
                {
                    "role": elem.get("role", ""),
                    "name": elem.get("name", ""),
                    "description": elem.get("description", ""),
                    "value": elem.get("value", ""),
                    "tag": elem.tag,
                    "attributes": dict(elem.attrib),
                }
            )

        return {
            "app_type": app_type,
            "app_name": app_name,
            "current_view": self._detect_current_view_basic(element_dicts),
            "key_elements": self._extract_key_elements_basic(element_dicts),
            "task_context": f"Application: {app_name}",
            "element_count": len(element_dicts),
        }

    def _detect_current_view_basic(self, elements: List[Dict[str, Any]]) -> str:
        """Detect current view based on element types (basic version)."""
        roles = [elem.get("role", "") for elem in elements]

        if "dialog" in roles:
            return "dialog_view"
        elif "menu" in roles:
            return "menu_view"
        elif "textbox" in roles and "button" in roles:
            return "form_view"
        elif "link" in roles:
            return "navigation_view"
        elif "heading" in roles:
            return "content_view"
        else:
            return "main_view"

    def _extract_key_elements_basic(self, elements: List[Dict[str, Any]]) -> List[str]:
        """Extract key elements for a specific application (basic version)."""
        key_elements = []

        # Prioritize interactive elements
        interactive_roles = ["button", "textbox", "link", "menu", "dialog"]

        for elem in elements:
            role = elem.get("role", "")
            if role in interactive_roles:
                name = elem.get("name", "")
                description = elem.get("description", "")

                element_desc = f"{role}"
                if name:
                    element_desc += f": {name}"
                if description:
                    element_desc += f" ({description})"

                key_elements.append(element_desc)

                # Limit to prevent overwhelming the LLM
                if len(key_elements) >= 10:
                    break

        return key_elements

    # ==================== HELPER METHODS ====================

    def _group_elements_by_app(self, root: ET.Element) -> Dict[str, List[ET.Element]]:
        """Group accessibility tree elements by application."""
        app_groups = defaultdict(list)
        parent_map = {child: parent for parent in root.iter() for child in parent}

        for elem in root.iter():
            app_name = self._get_app_name(elem, parent_map)
            if app_name and app_name != "unknown":
                app_groups[app_name].append(elem)

        return dict(app_groups)

    def _get_app_name(self, elem: ET.Element, parent_map: Dict) -> str:
        """Extract application name from element or parents."""
        current = elem
        visited = set()

        while current is not None and current not in visited:
            visited.add(current)

            app_name = current.get("application", "")
            if app_name:
                return app_name

            if current.tag in ["window", "frame", "application"]:
                name = current.get("name", "")
                if name:
                    return name

            current = parent_map.get(current)

        return "unknown"

    def _detect_app_type(self, app_name: str) -> str:
        """Detect application type from name."""
        app_lower = app_name.lower()

        if any(b in app_lower for b in ["chrome", "firefox", "safari", "edge", "browser"]):
            return "browser"
        elif "calc" in app_lower:
            return "libreoffice_calc"
        elif "writer" in app_lower:
            return "libreoffice_writer"
        elif "impress" in app_lower:
            return "libreoffice_impress"
        elif any(c in app_lower for c in ["code", "vscode"]):
            return "vs_code"
        elif "gimp" in app_lower:
            return "gimp"
        elif "vlc" in app_lower:
            return "vlc"
        elif any(f in app_lower for f in ["file", "manager", "explorer", "nautilus"]):
            return "file_manager"
        elif any(t in app_lower for t in ["terminal", "bash", "shell"]):
            return "terminal"
        elif any(s in app_lower for s in ["settings", "preferences", "system"]):
            return "system_settings"
        else:
            return "unknown"


if __name__ == "__main__":
    print("App State Extractor - Comprehensive UI information extraction")
    print("=" * 70)
    print("\nCapabilities:")
    print("✓ Browser: Full DOM extraction (buttons, links, forms, inputs, headings)")
    print("✓ LibreOffice: Document state via UNO (sheets, content, slides)")
    print("✓ All apps: Rich accessibility tree parsing")
    print("✓ Categorized elements: buttons, menus, text fields, checkboxes, etc.")
    print("✓ UI structure detection: menu bars, toolbars, dialogs, panels")
    print("✓ Interactive elements: Everything an LLM might want to manipulate")
    print("\n" + "=" * 70)
