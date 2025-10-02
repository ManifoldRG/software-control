"""
Comprehensive App State Extractor for Ubuntu AT-SPI2

Extracts LLM-consumable UI state for perturbation generation (CurriculumLLM, PerturbationLLM).

=== EXTRACTION STRATEGY BY TOOLKIT ===

1. Chromium/Electron (Chrome, VSCode):
   - Launch: ACCESSIBILITY_ENABLED=1 --force-renderer-accessibility
   - Extract: buttons[].{id, class, text, aria_label}, links[].{href, text},
              input_fields[].{name, type, placeholder}, forms[]
   - Usage: PerturbationLLM targets specific elements by ID/class for JS styling

2. LibreOffice (Calc, Writer, Impress):
   - Launch: Accessibility ON in Tools > Options > Accessibility
   - Extract: document_state.{sheets, active_sheet, sample_cells, slides},
              buttons[].{name}, menus[], text_fields[]
   - Usage: PerturbationLLM generates UNO code for grid/document theming

3. GTK/GNOME (Terminal, Nautilus, GIMP, VLC):
   - Launch: Ensure AT-SPI bus active (gsd-a11y-settings running)
   - Extract: interactive_elements[], ui_structure.{has_menu_bar, has_toolbar},
              visual_only_regions[] (canvas bounding boxes)
   - Usage: PerturbationLLM applies system-level bash commands

=== LLM CONSUMPTION PATTERN ===
CurriculumLLM: Uses app_states to understand available UI elements for scenario generation
PerturbationLLM: Uses app_states.buttons[], links[], inputs[] to target specific elements
                 References {app_states}.buttons[0].id in generated JavaScript/UNO code

=== ARCHITECTURE ===
VM Server (main.py): Generates AT-SPI XML via pyatspi → HTTP endpoint
Controller: Fetches XML from VM server
Extractor: Parses XML → Toolkit-aware extraction → LLM-friendly JSON
"""

import logging
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Any, Dict, List, Optional

# AT-SPI namespace definitions (matching accessibility_tree_handle.py)
ATTRIBUTES_NS_UBUNTU = "https://accessibility.windows.example.org/ns/attributes"
STATE_NS_UBUNTU = "https://accessibility.ubuntu.example.org/ns/state"
COMPONENT_NS_UBUNTU = "https://accessibility.ubuntu.example.org/ns/component"
VALUE_NS_UBUNTU = "https://accessibility.ubuntu.example.org/ns/value"
CLASS_NS_WINDOWS = "https://accessibility.windows.example.org/ns/class"

# UI element types for categorization
INTERACTIVE_ROLES = {
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
    "textfield",
    "textarea",
    "searchbox",
    "slider",
    "progress-bar",
}

UI_ELEMENT_TAGS = {
    "alert",
    "canvas",
    "check-box",
    "combo-box",
    "entry",
    "icon",
    "image",
    "paragraph",
    "scroll-bar",
    "section",
    "slider",
    "static",
    "table-cell",
    "terminal",
    "text",
    "netuiribbontab",
    "start",
    "trayclockwclass",
    "traydummysearchcontrol",
    "uiimage",
    "uiproperty",
    "uiribboncommandbar",
}


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

        Returns:
            List of app states with comprehensive UI information
        """
        a11y_tree = self.controller.get_accessibility_tree()
        if not a11y_tree:
            self.logger.warning("No accessibility tree available")
            return self._create_empty_state()

        try:
            root = ET.fromstring(a11y_tree)
            app_groups = self._group_elements_by_app(root)
            return self._process_app_groups(app_groups, use_comprehensive)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse accessibility tree: {e}")
            return self._create_empty_state()

    def _create_empty_state(self) -> List[Dict[str, Any]]:
        """Create empty state when no accessibility tree is available."""
        return [
            {
                "app_type": "unknown",
                "app_name": "unknown",
                "current_view": "unknown",
                "key_elements": [],
                "task_context": "No accessible applications detected",
                "element_count": 0,
            }
        ]

    def _process_app_groups(
        self, app_groups: Dict[str, List[ET.Element]], use_comprehensive: bool
    ) -> List[Dict[str, Any]]:
        """Process grouped elements into app states."""
        app_states = []

        for app_name, elements in app_groups.items():
            app_type = self._detect_app_type(app_name)
            if app_type == "unknown":
                continue

            app_state = self._extract_app_state(app_name, app_type, elements, use_comprehensive)
            app_states.append(app_state)

        return app_states if app_states else self._create_empty_state()

    def _extract_app_state(
        self, app_name: str, app_type: str, elements: List[ET.Element], use_comprehensive: bool
    ) -> Dict[str, Any]:
        """Extract state for a specific application."""
        if use_comprehensive:
            return self._extract_comprehensive_state(app_name, app_type, elements)
        else:
            return self._extract_basic_state(app_name, app_type, elements)

    def _extract_comprehensive_state(
        self, app_name: str, app_type: str, elements: List[ET.Element]
    ) -> Dict[str, Any]:
        """Extract comprehensive state with app-specific enhancements."""
        if app_type == "browser":
            return self._extract_browser_state(app_name, elements)
        elif app_type in ["libreoffice_calc", "libreoffice_writer", "libreoffice_impress"]:
            return self._extract_libreoffice_state(app_name, app_type, elements)
        else:
            return self._extract_generic_state(app_name, app_type, elements)

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

            # Format for LLM consumption (matches PerturbationLLM expectations)
            return {
                "dom_extracted": True,
                "page_url": dom_data["url"],
                "page_title": dom_data["title"],
                # LLM-accessible arrays (PerturbationLLM uses these directly)
                "buttons": dom_data["buttons"],  # [{id, class, text, aria_label, ...}]
                "links": dom_data["links"],  # [{href, text, id, class, ...}]
                "input_fields": dom_data["inputs"],  # [{name, type, placeholder, value, ...}]
                "forms": dom_data["forms"],  # [{id, class, fields: [...]}]
                "headings": dom_data["headings"],  # [{level, text}]
                "images": dom_data["images"],  # [{alt, src, title}]
                "content_sections": dom_data["content_sections"],
                # Summary for LLM context
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
        Extract LibreOffice state using VCL/Java Bridge accessibility.

        Strategy: Focus on deep semantic data model via Table/Document interfaces
        - Calc: Query Atspi.Table interface for cell access by logical coordinates
        - Writer/Impress: Traverse Atspi.Document structure for logical organization

        Note: Requires accessibility enabled in Tools > Options > Accessibility
        """
        self.logger.info(f"Extracting LibreOffice state for {app_name}")

        # Base state from accessibility tree
        base_state = self._extract_generic_state(app_name, app_type, elements)

        # Extract toolkit-specific semantic data
        if app_type == "libreoffice_calc":
            table_data = self._extract_calc_table_structure(elements)
            base_state.update(table_data)
        elif app_type in ["libreoffice_writer", "libreoffice_impress"]:
            document_data = self._extract_document_structure(elements)
            base_state.update(document_data)

        # Try to get additional document-specific information via UNO
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

    def _extract_calc_table_structure(self, elements: List[ET.Element]) -> Dict[str, Any]:
        """
        Extract Calc-specific table structure using Table interface metadata.

        Focuses on logical cell coordinates (row/col) rather than pixel positions
        for stable data manipulation grounding.

        Note: LibreOffice Calc cells use cell names (e.g., "D1", "A2") rather than
        explicit row/column attributes.
        """
        table_structure = {
            "table_cells": [],
            "table_metadata": {
                "rows_visible": 0,
                "columns_visible": 0,
                "has_table_interface": False,
            },
        }

        def parse_cell_name(cell_name: str) -> tuple:
            """
            Parse cell name like 'D1' into (column, row).
            Returns ('D', 1) for 'D1', ('AA', 10) for 'AA10', etc.
            """
            if not cell_name:
                return (None, None)

            # Extract column letters and row number
            col = ""
            row = ""
            for char in cell_name:
                if char.isalpha():
                    col += char
                elif char.isdigit():
                    row += char

            try:
                return (col, int(row)) if col and row else (None, None)
            except ValueError:
                return (None, None)

        # Look for table-cell elements directly (they may not have a parent table element)
        for elem in elements:
            # Check if this is a table element or contains table cells
            if elem.tag.endswith("table") or elem.get("role", "") == "table":
                table_structure["table_metadata"]["has_table_interface"] = True

            # Extract table cells (check tag directly)
            if elem.tag.endswith("table-cell") or "table-cell" in elem.tag:
                table_structure["table_metadata"]["has_table_interface"] = True

                # Get cell name (e.g., "D1", "A2")
                cell_name = elem.get("name", "")
                cell_value = elem.get(f"{{{VALUE_NS_UBUNTU}}}value", "")
                cell_text = elem.text if elem.text else ""
                formula = elem.get(f"{{{ATTRIBUTES_NS_UBUNTU}}}Formula", "")

                # Parse cell coordinates from name
                col, row = parse_cell_name(cell_name)

                if col and row:
                    cell_data = {
                        "name": cell_name,
                        "column": col,
                        "row": row,
                        "text": cell_text,
                        "value": cell_value,
                        "formula": formula if formula else None,
                    }
                    table_structure["table_cells"].append(cell_data)

        # Calculate visible rows and columns
        if table_structure["table_cells"]:
            unique_cols = {c["column"] for c in table_structure["table_cells"] if c["column"]}
            unique_rows = {c["row"] for c in table_structure["table_cells"] if c["row"]}
            table_structure["table_metadata"]["columns_visible"] = len(unique_cols)
            table_structure["table_metadata"]["rows_visible"] = len(unique_rows)

        return table_structure

    def _extract_document_structure(self, elements: List[ET.Element]) -> Dict[str, Any]:
        """
        Extract Writer/Impress document structure via Document interface.

        Focuses on logical organizational components (sections, paragraphs)
        rather than pixel-based navigation.
        """
        document_structure = {
            "sections": [],
            "paragraphs": [],
            "slides": [],  # For Impress
            "has_document_interface": False,
        }

        for elem in elements:
            # Look for document structure elements
            if elem.get("role", "") in ["document", "document-frame", "section"]:
                document_structure["has_document_interface"] = True

            if elem.tag.endswith("section"):
                document_structure["sections"].append(
                    {
                        "name": elem.get("name", ""),
                        "text_preview": (elem.text if elem.text else "")[:200],
                    }
                )

            if elem.tag.endswith("paragraph"):
                document_structure["paragraphs"].append(
                    {
                        "text": elem.text if elem.text else "",
                        "position": elem.get(f"{{{COMPONENT_NS_UBUNTU}}}screencoord", ""),
                    }
                )

            # For Impress slides
            if elem.tag.endswith("slide") or "slide" in elem.get("role", ""):
                document_structure["slides"].append(
                    {
                        "name": elem.get("name", ""),
                        "index": elem.get(f"{{{ATTRIBUTES_NS_UBUNTU}}}index", ""),
                    }
                )

        return document_structure

    # ==================== GENERIC STATE EXTRACTION ====================

    def _extract_generic_state(
        self, app_name: str, app_type: str, elements: List[ET.Element]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive state from accessibility tree.

        This works for ALL applications (GTK apps, Electron apps, etc.)
        Applies toolkit-specific strategies:
        - GTK/GNOME: Standard controls, text grids, spatial data
        - Chromium/Electron: Semantic/hierarchical pruning
        - Detects visual-only regions (canvas) that require multimodal LLM analysis
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

        # Detect visual-only regions (canvas objects in GIMP, VLC, etc.)
        visual_only_regions = self._detect_visual_only_regions(parsed_elements, app_type)

        # Create linearized accessibility tree
        linearized_tree = self._create_linearized_accessibility_tree(parsed_elements)

        state = {
            "app_type": app_type,
            "app_name": app_name,
            "current_view": current_view,
            "element_count": len(parsed_elements),
            # Categorized elements - LLM-accessible arrays
            "buttons": categorized.get("buttons", []),
            "text_fields": categorized.get("text_fields", []),
            "labels": categorized.get("labels", []),
            "menus": categorized.get("menus", []),
            "menu_items": categorized.get("menu_items", []),
            "checkboxes": categorized.get("checkboxes", []),
            "radio_buttons": categorized.get("radio_buttons", []),
            "combo_boxes": categorized.get("combo_boxes", []),
            "tabs": categorized.get("tabs", []),
            "panels": categorized.get("panels", []),
            "scrollbars": categorized.get("scrollbars", []),
            "tables": categorized.get("tables", []),
            "links": categorized.get("links", []),
            "images": categorized.get("images", []),
            "headings": categorized.get("headings", []),
            # UI structure
            "ui_structure": ui_structure,
            "interactive_elements": interactive_elements[:50],  # Limit for LLM
            "active_dialogs": active_dialogs,
            "menu_structure": menu_structure,
            "visual_only_regions": visual_only_regions,
            # Linearized accessibility tree for LLM consumption
            "linearized_accessibility_tree": linearized_tree,
            # Summary statistics
            "summary": {
                "total_buttons": len(categorized.get("buttons", [])),
                "total_text_fields": len(categorized.get("text_fields", [])),
                "total_menus": len(categorized.get("menus", [])),
                "total_menu_items": len(categorized.get("menu_items", [])),
                "total_links": len(categorized.get("links", [])),
                "total_images": len(categorized.get("images", [])),
                "total_headings": len(categorized.get("headings", [])),
                "total_interactive": len(interactive_elements),
                "has_active_dialog": len(active_dialogs) > 0,
                "has_visual_only_regions": len(visual_only_regions) > 0,
            },
        }

        # Ensure all arrays are properly initialized (prevents KeyError in LLM code)
        for key in ["buttons", "links", "text_fields", "input_fields", "forms", "menus", "menu_items"]:
            if key not in state:
                state[key] = []

        return state

    def _parse_accessibility_elements(self, elements: List[ET.Element]) -> List[Dict[str, Any]]:
        """
        Parse accessibility tree elements into structured dictionaries.

        Enhanced with proper AT-SPI namespace handling and better text extraction.
        """
        parsed = []

        for elem in elements:
            # Extract all attributes with proper namespace handling
            element_data = {
                "tag": elem.tag,
                "role": elem.get("role", ""),
                "name": elem.get("name", ""),
                "description": elem.get("description", ""),
                "value": elem.get("value", ""),
                "text": elem.text if elem.text else "",
                "attributes": {},
            }

            # Extract namespaced attributes with proper AT-SPI namespaces
            for key, value in elem.attrib.items():
                if "}" in key:
                    namespace, attr_name = key.split("}", 1)
                    namespace = namespace.lstrip("{")

                    # Map to friendly names based on actual AT-SPI namespaces
                    if namespace == COMPONENT_NS_UBUNTU:
                        element_data["attributes"][f"component_{attr_name}"] = value
                    elif namespace == STATE_NS_UBUNTU:
                        element_data["attributes"][f"state_{attr_name}"] = value
                    elif namespace == ATTRIBUTES_NS_UBUNTU:
                        element_data["attributes"][f"attr_{attr_name}"] = value
                    elif namespace == VALUE_NS_UBUNTU:
                        element_data["attributes"][f"value_{attr_name}"] = value
                    else:
                        element_data["attributes"][attr_name] = value
                else:
                    element_data["attributes"][key] = value

            # Extract coordinates with proper parsing
            screencoord = element_data["attributes"].get("component_screencoord", "")
            size = element_data["attributes"].get("component_size", "")

            if screencoord and size:
                try:
                    # Parse coordinates like "(x, y)" format
                    coords_match = re.match(r"\((\d+),\s*(\d+)\)", screencoord)
                    size_match = re.match(r"\((\d+),\s*(\d+)\)", size)

                    if coords_match and size_match:
                        x, y = int(coords_match.group(1)), int(coords_match.group(2))
                        w, h = int(size_match.group(1)), int(size_match.group(2))

                        element_data["position"] = {
                            "x": x,
                            "y": y,
                            "width": w,
                            "height": h,
                            "center_x": x + w // 2,
                            "center_y": y + h // 2,
                        }
                except Exception as e:
                    self.logger.debug(f"Error parsing coordinates: {e}")

            # Enhanced visibility and state checking
            element_data["visible"] = element_data["attributes"].get("state_visible", "") == "true"
            element_data["showing"] = element_data["attributes"].get("state_showing", "") == "true"
            element_data["enabled"] = element_data["attributes"].get("state_enabled", "") == "true"
            element_data["focused"] = element_data["attributes"].get("state_focused", "") == "true"
            element_data["editable"] = element_data["attributes"].get("state_editable", "") == "true"
            element_data["checkable"] = element_data["attributes"].get("state_checkable", "") == "true"
            element_data["checked"] = element_data["attributes"].get("state_checked", "") == "true"
            element_data["selected"] = element_data["attributes"].get("state_selected", "") == "true"
            element_data["expandable"] = element_data["attributes"].get("state_expandable", "") == "true"
            element_data["expanded"] = element_data["attributes"].get("state_expanded", "") == "true"

            # Enhanced text extraction (matching accessibility_tree_handle.py logic)
            text = element_data["text"]
            name = element_data["name"]

            if not text and name:
                text = name
            elif name and text and text != name:
                text = f"{name} ({text})"

            # Clean up text (remove Unicode replacement characters)
            text = text.replace("\ufffc", "").replace("\ufffd", "").strip()
            element_data["text"] = text

            # Extract value information
            value = element_data["attributes"].get("value_value", "")
            if value:
                element_data["value"] = value

            parsed.append(element_data)

        return parsed

    def _categorize_elements(self, elements: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Categorize elements by type for easy access with enhanced AT-SPI support."""
        categories = defaultdict(list)

        for elem in elements:
            if not self._is_element_visible(elem):
                continue

            category = self._determine_element_category(elem)
            if category:
                categories[category].append(self._create_element_summary(elem, category))

        return dict(categories)

    def _is_element_visible(self, elem: Dict[str, Any]) -> bool:
        """
        Check if element is visible and showing.

        More lenient: element is considered visible if it has EITHER visible=true OR showing=true.
        This is important for LibreOffice where many elements only have one flag set.
        """
        return elem.get("visible", False) or elem.get("showing", False)

    def _determine_element_category(self, elem: Dict[str, Any]) -> Optional[str]:
        """Determine the category of an element based on role and tag."""
        role = elem["role"].lower()
        tag = elem["tag"].lower()

        # Define category mapping rules
        category_rules = {
            "buttons": lambda r, t: ("button" in r or "push-button" in r or t.endswith("button")),
            "text_fields": lambda r, t: (
                r in ["text", "entry", "text-box"] or t.endswith("textfield") or t.endswith("textarea")
            ),
            "labels": lambda r, t: (r == "label" or t.endswith("label")),
            "menus": lambda r, t: (r == "menu-bar" or t.endswith("menu-bar")),
            "menu_items": lambda r, t: (r == "menu-item" or t.endswith("menu-item")),
            "checkboxes": lambda r, t: (r == "check-box" or t.endswith("check-box")),
            "radio_buttons": lambda r, t: (r == "radio-button" or t.endswith("radio-button")),
            "combo_boxes": lambda r, t: (r == "combo-box" or t.endswith("combo-box")),
            "tabs": lambda r, t: (r == "tab" or t.endswith("tab")),
            "panels": lambda r, t: (r == "panel" or t == "scroll-pane" or t.endswith("panel")),
            "scrollbars": lambda r, t: (r == "scroll-bar" or t.endswith("scroll-bar")),
            "tables": lambda r, t: (r == "table" or t.endswith("table")),
            "links": lambda r, t: (r == "link" or t.endswith("link")),
            "images": lambda r, t: (r == "image" or t.endswith("image") or t == "image"),
            "headings": lambda r, t: (r == "heading" or t.endswith("heading")),
        }

        for category, rule in category_rules.items():
            if rule(role, tag):
                return category

        return None

    def _create_element_summary(self, elem: Dict[str, Any], category: str) -> Dict[str, Any]:
        """Create a summary of an element for its category."""
        base_summary = {
            "name": elem["name"],
            "text": elem["text"],
            "position": elem.get("position", {}),
        }

        # Add category-specific fields
        if category in [
            "buttons",
            "text_fields",
            "menu_items",
            "checkboxes",
            "radio_buttons",
            "combo_boxes",
            "links",
        ]:
            base_summary.update(
                {
                    "enabled": elem["enabled"],
                    "focused": elem["focused"],
                }
            )

        if category == "text_fields":
            base_summary.update(
                {
                    "value": elem["value"],
                    "placeholder": elem["attributes"].get("attr_placeholder", ""),
                    "editable": elem["editable"],
                }
            )

        if category == "checkboxes":
            base_summary.update(
                {
                    "checked": elem["checked"],
                    "checkable": elem["checkable"],
                }
            )

        if category == "radio_buttons":
            base_summary.update(
                {
                    "selected": elem["selected"],
                }
            )

        if category == "links":
            base_summary.update(
                {
                    "href": elem["attributes"].get("attr_href", ""),
                }
            )

        if category == "images":
            base_summary.update(
                {
                    "alt": elem["attributes"].get("attr_alt", ""),
                }
            )

        if category == "headings":
            base_summary.update(
                {
                    "level": elem["attributes"].get("attr_level", ""),
                }
            )

        if category == "scrollbars":
            base_summary.update(
                {
                    "orientation": "vertical"
                    if "vertical" in elem["attributes"].get("attr_orientation", "")
                    else "horizontal",
                }
            )

        if category == "tables":
            base_summary.update(
                {
                    "rows": elem["attributes"].get("attr_rows", ""),
                    "columns": elem["attributes"].get("attr_columns", ""),
                }
            )

        return base_summary

    def _extract_interactive_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract ALL interactive elements that LLM might want to manipulate."""
        interactive = []

        for elem in elements:
            if not self._is_element_visible(elem) or not self._is_element_interactive(elem):
                continue

            interactive.append(self._create_interactive_element_summary(elem))

        return interactive

    def _is_element_interactive(self, elem: Dict[str, Any]) -> bool:
        """Check if element is interactive based on role and tag."""
        role = elem["role"].lower()
        tag = elem["tag"].lower()

        return (
            any(r in role for r in INTERACTIVE_ROLES)
            or any(tag.endswith(t) for t in ["button", "textfield", "textarea", "link", "tab", "menu-item"])
            or tag in ["entry", "combo-box", "check-box", "radio-button", "slider"]
        )

    def _create_interactive_element_summary(self, elem: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary for an interactive element."""
        return {
            "type": elem["role"] or elem["tag"],
            "name": elem["name"],
            "text": elem["text"],
            "description": elem["description"],
            "value": elem["value"],
            "position": elem.get("position", {}),
            "focused": elem["focused"],
            "enabled": elem["enabled"],
            "editable": elem.get("editable", False),
            "checked": elem.get("checked", False),
            "selected": elem.get("selected", False),
        }

    def _create_linearized_accessibility_tree(self, elements: List[Dict[str, Any]]) -> str:
        """
        Create a linearized accessibility tree similar to accessibility_tree_handle.py.

        This provides a clean, tabular format that's easy for LLMs to parse.
        """
        lines = ["tag\ttext\tposition (center x & y)\tsize (w & h)"]

        for elem in elements:
            if not elem.get("visible", False) or not elem.get("showing", False):
                continue

            tag = elem["tag"]
            text = elem["text"]
            position = elem.get("position", {})

            if position:
                center_x = position.get("center_x", 0)
                center_y = position.get("center_y", 0)
                width = position.get("width", 0)
                height = position.get("height", 0)

                pos_str = f"({center_x}, {center_y})"
                size_str = f"({width}, {height})"
            else:
                pos_str = "(0, 0)"
                size_str = "(0, 0)"

            # Clean text for tabular format
            text = text.replace("\n", "\\n").replace("\t", " ")

            lines.append(f"{tag}\t{text}\t{pos_str}\t{size_str}")

        return "\n".join(lines)

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
        visible_roles = [elem["role"].lower() for elem in elements if elem.get("visible", False)]

        view_detectors = [
            (lambda roles: any("dialog" in r for r in roles), self._detect_dialog_view),
            (
                lambda roles: any("menu" in r for r in roles) and not any("menu-bar" in r for r in roles),
                lambda: "menu_expanded",
            ),
            (lambda roles: sum("text" in r or "entry" in r for r in roles) > 3, lambda: "form_view"),
            (lambda roles: any("table" in r for r in roles), lambda: "table_view"),
            (lambda roles: sum("tab" in r for r in roles) > 2, lambda: "tabbed_view"),
        ]

        for condition, detector in view_detectors:
            if condition(visible_roles):
                return detector(elements) if callable(detector) else detector

        return "main_view"

    def _detect_dialog_view(self, elements: List[Dict[str, Any]]) -> str:
        """Detect dialog view with specific dialog names."""
        dialog_names = [
            elem["name"]
            for elem in elements
            if "dialog" in elem["role"].lower() and elem.get("visible", False)
        ]
        return f"dialog_view ({', '.join(dialog_names[:2])})"

    def _detect_active_dialogs(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect any active dialogs with their content."""
        return [
            {
                "name": elem["name"],
                "description": elem["description"],
                "modal": elem["attributes"].get("state_modal", "") == "true",
            }
            for elem in elements
            if "dialog" in elem["role"].lower() and elem.get("visible", False)
        ]

    def _extract_menu_structure(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract menu structure for understanding available actions."""
        return [
            {"name": elem["name"], "type": elem["role"], "enabled": elem["enabled"]}
            for elem in elements
            if ("menu-bar" in elem["role"].lower() or "menu" in elem["role"].lower())
            and elem.get("visible", False)
        ]

    def _detect_visual_only_regions(
        self, elements: List[Dict[str, Any]], app_type: str
    ) -> List[Dict[str, Any]]:
        """
        Detect visual-only regions (canvas objects) that require multimodal analysis.

        Strategy: Canvas objects in GIMP (drawing area) and VLC (video display)
        are opaque - they have no informative accessible children. These regions
        must be flagged for screenshot-based analysis by multimodal LLMs.

        Args:
            elements: Parsed accessibility elements
            app_type: Application type (e.g., "vlc", "gimp")

        Returns list of bounding boxes for visual-only regions.
        """
        visual_regions = []

        # Check if this is a media player or graphics app
        is_media_player = app_type in ["vlc", "media_player"]
        is_graphics_app = app_type in ["gimp", "image_editor"]

        # Debug: Log element details for media players/graphics apps
        if is_media_player or is_graphics_app:
            self.logger.info(f"Detecting visual regions for {app_type}, total elements: {len(elements)}")
            for elem in elements[:20]:  # Log first 20 elements
                self.logger.debug(
                    f"  Element: role={elem['role']}, tag={elem['tag']}, name={elem.get('name', '')[:30]}, "
                    f"visible={elem.get('visible', False)}, showing={elem.get('showing', False)}, "
                    f"position={elem.get('position', {})}"
                )

        # Track all potential canvas/video areas for fallback
        potential_regions = []

        for elem in elements:
            role = elem["role"].lower()
            tag = elem["tag"].lower()
            name = elem.get("name", "").lower()

            # Detect canvas elements (GIMP drawing area, VLC video display)
            is_explicit_canvas = (
                role == "canvas"
                or tag == "canvas"
                or "canvas" in role
                or "drawing-area" in tag
                or "drawing-area" in role
                or "video" in role
                or "video" in name
            )

            # For media players/graphics apps, large unnamed panels/frames are likely content areas
            is_potential_content_area = (
                (is_media_player or is_graphics_app)
                and role in ["panel", "frame", "layered-pane", "root-pane", "scroll-pane"]
                and not name  # Empty name suggests content area rather than UI chrome
            )

            if (is_explicit_canvas or is_potential_content_area) and (
                elem.get("visible", False) or elem.get("showing", False)
            ):
                position = elem.get("position", {})
                if position and position.get("width", 0) > 100 and position.get("height", 0) > 100:
                    area = position.get("width", 0) * position.get("height", 0)
                    visual_region = {
                        "type": "visual_only_canvas" if is_explicit_canvas else "potential_content_area",
                        "role": role,
                        "tag": tag,
                        "name": elem["name"],
                        "bounding_box": {
                            "x": position.get("x", 0),
                            "y": position.get("y", 0),
                            "width": position.get("width", 0),
                            "height": position.get("height", 0),
                            "center_x": position.get("center_x", 0),
                            "center_y": position.get("center_y", 0),
                        },
                        "note": "This region requires screenshot-based multimodal LLM analysis - no AT-SPI children available",
                    }

                    if is_explicit_canvas:
                        visual_regions.append(visual_region)
                    else:
                        potential_regions.append((area, visual_region))

        # For media players/graphics apps, if no explicit canvas found, use largest potential region
        if (is_media_player or is_graphics_app) and not visual_regions and potential_regions:
            # Sort by area and take largest (likely the video/canvas area)
            potential_regions.sort(key=lambda x: x[0], reverse=True)
            if potential_regions[0][0] > 50000:  # At least ~224x224 pixels
                largest_region = potential_regions[0][1]
                largest_region["type"] = "visual_only_canvas"  # Promote to canvas
                visual_regions.append(largest_region)

        # Final fallback: If VLC/GIMP have very few elements and no canvas detected,
        # assume the entire window is a visual-only region (common for apps with poor AT-SPI support)
        if (is_media_player or is_graphics_app) and not visual_regions and len(elements) < 20:
            self.logger.info(
                f"{app_type} has only {len(elements)} accessible elements and no detected canvas - "
                "assuming entire window is visual-only region"
            )
            # Create a generic visual region note for the LLM
            visual_regions.append(
                {
                    "type": "visual_only_canvas",
                    "role": "window",
                    "tag": "window",
                    "name": app_type,
                    "bounding_box": {
                        "x": 0,
                        "y": 0,
                        "width": 0,
                        "height": 0,
                        "center_x": 0,
                        "center_y": 0,
                    },
                    "note": (
                        f"This {app_type} application has minimal AT-SPI accessibility tree exposure. "
                        "The main content area (video player, canvas, etc.) is not exposed via accessibility APIs. "
                        "Screenshot-based multimodal LLM analysis is REQUIRED for understanding the visual content."
                    ),
                }
            )

        return visual_regions

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
        """Group accessibility tree elements by application using enhanced filtering."""
        app_groups = defaultdict(list)
        parent_map = {child: parent for parent in root.iter() for child in parent}

        # First, find active applications (matching accessibility_tree_handle.py logic)
        active_apps = self._find_active_applications(root)

        for elem in root.iter():
            app_name = self._get_app_name(elem, parent_map)
            if app_name and app_name != "unknown" and app_name in active_apps:
                # Apply node filtering similar to accessibility_tree_handle.py
                if self._judge_node(elem):
                    app_groups[app_name].append(elem)

        return dict(app_groups)

    def _find_active_applications(self, root: ET.Element) -> List[str]:
        """Find active applications (matching accessibility_tree_handle.py logic)."""
        apps_with_active_tag = []
        frame_names_with_active_tag = []

        for application in list(root):
            app_name = application.get("name")
            if not app_name:
                continue

            for frame in application:
                is_active = frame.get(f"{{{STATE_NS_UBUNTU}}}active", "false") == "true"
                if is_active:
                    apps_with_active_tag.append(app_name)
                    # Also include frame name for apps like LibreOffice
                    # where frame name is "Invoice.xlsx - LibreOffice Calc"
                    frame_name = frame.get("name", "")
                    if frame_name:
                        frame_names_with_active_tag.append(frame_name)

        if apps_with_active_tag:
            # Return both app names and frame names
            return apps_with_active_tag + frame_names_with_active_tag + ["gnome-shell"]
        else:
            return ["gjs", "gnome-shell"]

    def _judge_node(self, node: ET.Element, check_image: bool = True) -> bool:
        """Judge if a node should be kept (matching accessibility_tree_handle.py logic)."""
        # Check if it's a relevant UI element
        keeps = (
            node.tag.startswith("document")
            or node.tag.endswith("item")
            or node.tag.endswith("button")
            or node.tag.endswith("heading")
            or node.tag.endswith("label")
            or node.tag.endswith("scrollbar")
            or node.tag.endswith("searchbox")
            or node.tag.endswith("textbox")
            or node.tag.endswith("link")
            or node.tag.endswith("tabelement")
            or node.tag.endswith("textfield")
            or node.tag.endswith("textarea")
            or node.tag.endswith("menu")
            or node.tag
            in {
                "alert",
                "canvas",
                "check-box",
                "combo-box",
                "entry",
                "icon",
                "image",
                "paragraph",
                "scroll-bar",
                "section",
                "slider",
                "static",
                "table-cell",
                "terminal",
                "text",
                "netuiribbontab",
                "start",
                "trayclockwclass",
                "traydummysearchcontrol",
                "uiimage",
                "uiproperty",
                "uiribboncommandbar",
            }
        )

        # Check visibility and showing states
        # More lenient: accept if EITHER showing OR visible is true
        # This is important for LibreOffice where many elements only have one flag
        keeps = (
            keeps
            and (
                node.get(f"{{{STATE_NS_UBUNTU}}}showing", "false") == "true"
                or node.get(f"{{{STATE_NS_UBUNTU}}}visible", "false") == "true"
            )
            and (
                node.get("name", "") != ""
                or (node.text is not None and len(node.text) > 0)
                or (check_image and node.get("image", "false") == "true")
            )
        )

        # Check coordinates and size
        try:
            screencoord = node.get(f"{{{COMPONENT_NS_UBUNTU}}}screencoord", "(-1, -1)")
            size = node.get(f"{{{COMPONENT_NS_UBUNTU}}}size", "(-1, -1)")
            coords = eval(screencoord)
            sizes = eval(size)
            keeps = keeps and coords[0] >= 0 and coords[1] >= 0 and sizes[0] > 0 and sizes[1] > 0
        except Exception:
            keeps = False

        return keeps

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
        """
        Detect application type from name.

        Args:
            app_name: Can be application name ("soffice") or frame name
                     ("Invoice.xlsx - LibreOffice Calc")
        """
        app_lower = app_name.lower()

        # Browser detection
        if any(b in app_lower for b in ["chrome", "firefox", "safari", "edge", "browser"]):
            return "browser"

        # LibreOffice detection - check for both app name and frame name patterns
        # Frame names: "Document.xlsx - LibreOffice Calc", "Report.odt - LibreOffice Writer"
        elif "calc" in app_lower or (
            ("libreoffice" in app_lower or "soffice" in app_lower)
            and (".xlsx" in app_lower or ".ods" in app_lower or ".xls" in app_lower or ".csv" in app_lower)
        ):
            return "libreoffice_calc"
        elif "writer" in app_lower or (
            ("libreoffice" in app_lower or "soffice" in app_lower)
            and (".odt" in app_lower or ".doc" in app_lower or ".docx" in app_lower)
        ):
            return "libreoffice_writer"
        elif "impress" in app_lower or (
            ("libreoffice" in app_lower or "soffice" in app_lower)
            and (".odp" in app_lower or ".ppt" in app_lower or ".pptx" in app_lower)
        ):
            return "libreoffice_impress"

        # Other applications
        elif any(c in app_lower for c in ["code", "vscode"]):
            return "vs_code"
        elif "gimp" in app_lower:
            return "gimp"
        elif "vlc" in app_lower:
            return "vlc"
        elif any(f in app_lower for f in ["file", "manager", "explorer", "nautilus"]):
            return "file_manager"
        elif any(t in app_lower for t in ["terminal", "bash", "shell"]) and "gnome-shell" not in app_lower:
            return "terminal"
        elif any(s in app_lower for s in ["settings", "preferences", "system"]):
            return "system_settings"
        else:
            return "unknown"


if __name__ == "__main__":
    print("App State Extractor - Toolkit-Aware AT-SPI2 Extraction for Ubuntu")
    print("=" * 80)
    print("\nToolkit-Specific Strategies:")
    print("\n1. Chromium/Electron (Chrome, VSCode):")
    print("   ✓ Semantic/hierarchical DOM node pruning")
    print("   ✓ Rich role, name, state, bounding_box extraction")
    print("   ✓ Text extraction via Atspi.Text interface")
    print("   ⚠ Requires: ACCESSIBILITY_ENABLED=1 + --force-renderer-accessibility")

    print("\n2. LibreOffice Suite (Calc, Writer, Impress):")
    print("   ✓ Deep semantic data model via Table/Document interfaces")
    print("   ✓ Logical cell coordinates (row/col) for Calc")
    print("   ✓ Document structure extraction for Writer/Impress")
    print("   ✓ UNO API integration for additional state")
    print("   ⚠ Requires: Accessibility enabled in Tools > Options")

    print("\n3. GTK/GNOME Native (Terminal, Nautilus, GIMP):")
    print("   ✓ Standard controls and text grids")
    print("   ✓ Precise spatial data extraction")
    print("   ✓ Visual-only region detection (canvas objects)")
    print("   ⚠ Requires: AT-SPI bus active (e.g., Orca running)")

    print("\nGeneral Capabilities:")
    print("✓ Categorized elements: buttons, menus, text fields, checkboxes, links")
    print("✓ UI structure detection: menu bars, toolbars, dialogs, panels")
    print("✓ Interactive elements: Everything an LLM might want to manipulate")
    print("✓ AT-SPI namespace handling: Proper Ubuntu accessibility tree parsing")
    print("✓ Enhanced filtering: Based on accessibility_tree_handle.py logic")
    print("✓ Linearized trees: Clean tabular format for LLM consumption")
    print("✓ Visual-only regions: Flags canvas areas needing multimodal analysis")
    print("\n" + "=" * 80)
