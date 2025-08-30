"""Component grouping logic for identifying functional UI components."""

import logging
from dataclasses import dataclass
from typing import Any

from perturbation_engine.sampling.ui_visual_design.data_types import (
    ComponentAttribute,
    ComponentType,
    Element,
    FunctionalComponent,
)


@dataclass
class GroupingRule:
    """Rule for grouping elements into functional components."""

    name: str
    component_type: ComponentType
    conditions: list[dict[str, Any]]
    attribute_extractors: list[dict[str, Any]] = None


class ComponentGrouper:
    """Groups individual DOM elements into functional UI components."""

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._grouping_rules = self._initialize_grouping_rules()

    def group_elements(self, elements: list[Element]) -> list[FunctionalComponent]:
        """Group elements into functional components using rule-based logic."""
        self._logger.info(f"Grouping {len(elements)} elements into functional components")

        components = []
        processed_elements = set()

        # First pass: identify standalone components
        for element in elements:
            if element.element_id in processed_elements:
                continue

            component = self._identify_standalone_component(element, elements)
            if component:
                components.append(component)
                processed_elements.update(e.element_id for e in component.elements)

        # Second pass: identify composite components
        remaining_elements = [e for e in elements if e.element_id not in processed_elements]
        composite_components = self._identify_composite_components(remaining_elements)
        components.extend(composite_components)

        self._logger.info(f"Identified {len(components)} functional components")
        return components

    def _initialize_grouping_rules(self) -> list[GroupingRule]:
        """Initialize the rules for component identification."""
        return [
            # Button components
            GroupingRule(
                name="button",
                component_type=ComponentType.BUTTON,
                conditions=[
                    {"element_type": "button"},
                    {"element_type": "input", "attributes": {"type": ["button", "submit", "reset"]}},
                    {"attributes": {"role": "button"}},
                    {"attributes": {"onclick": "not_none"}},
                ],
                attribute_extractors=[
                    {"name": "text", "extractor": "text_content"},
                    {"name": "type", "extractor": "input_type"},
                    {"name": "disabled", "extractor": "is_disabled"},
                ],
            ),
            # Input field components
            GroupingRule(
                name="input_field",
                component_type=ComponentType.INPUT_FIELD,
                conditions=[
                    {
                        "element_type": "input",
                        "attributes": {"type": ["text", "email", "password", "search", "tel", "url"]},
                    },
                    {"element_type": "textarea"},
                ],
                attribute_extractors=[
                    {"name": "placeholder", "extractor": "placeholder"},
                    {"name": "type", "extractor": "input_type"},
                    {"name": "required", "extractor": "is_required"},
                    {"name": "value", "extractor": "current_value"},
                ],
            ),
            # Dropdown components
            GroupingRule(
                name="dropdown",
                component_type=ComponentType.DROPDOWN,
                conditions=[
                    {"element_type": "select"},
                    {"element_type": "input", "attributes": {"list": "not_none"}},
                ],
                attribute_extractors=[
                    {"name": "options", "extractor": "dropdown_options"},
                    {"name": "selected", "extractor": "selected_option"},
                    {"name": "multiple", "extractor": "is_multiple"},
                ],
            ),
            # Search bar components
            GroupingRule(
                name="search_bar",
                component_type=ComponentType.SEARCH_BAR,
                conditions=[
                    {"element_type": "input", "attributes": {"type": "search"}},
                    {"attributes": {"placeholder": "search_pattern"}},
                    {"attributes": {"aria-label": "search_pattern"}},
                ],
                attribute_extractors=[
                    {"name": "placeholder", "extractor": "placeholder"},
                    {"name": "search_button", "extractor": "adjacent_button"},
                ],
            ),
            # Navigation components
            GroupingRule(
                name="navigation",
                component_type=ComponentType.NAVIGATION,
                conditions=[
                    {"element_type": "nav"},
                    {"attributes": {"role": "navigation"}},
                    {"attributes": {"aria-label": "nav_pattern"}},
                ],
                attribute_extractors=[
                    {"name": "links", "extractor": "nav_links"},
                    {"name": "current_page", "extractor": "current_nav_item"},
                ],
            ),
            # Form components
            GroupingRule(
                name="form",
                component_type=ComponentType.FORM,
                conditions=[
                    {"element_type": "form"},
                ],
                attribute_extractors=[
                    {"name": "action", "extractor": "form_action"},
                    {"name": "method", "extractor": "form_method"},
                    {"name": "fields", "extractor": "form_fields"},
                ],
            ),
        ]

    def _identify_standalone_component(
        self, element: Element, all_elements: list[Element]
    ) -> FunctionalComponent | None:
        """Identify if an element is a standalone component."""
        for rule in self._grouping_rules:
            if self._matches_rule(element, rule):
                return self._create_component(element, rule, all_elements)
        return None

    def _identify_composite_components(self, elements: list[Element]) -> list[FunctionalComponent]:
        """Identify composite components that span multiple elements."""
        components = []

        # Group by proximity and semantic relationships
        # This is a simplified version - in practice, you'd want more sophisticated spatial analysis

        # Look for input + label pairs
        input_label_groups = self._group_input_labels(elements)
        components.extend(input_label_groups)

        # Look for button groups
        button_groups = self._group_buttons(elements)
        components.extend(button_groups)

        # Look for card-like structures
        card_groups = self._group_cards(elements)
        components.extend(card_groups)

        return components

    def _matches_rule(self, element: Element, rule: GroupingRule) -> bool:
        """Check if an element matches a grouping rule."""
        for condition in rule.conditions:
            if self._matches_condition(element, condition):
                return True
        return False

    def _matches_condition(self, element: Element, condition: dict[str, Any]) -> bool:
        """Check if an element matches a specific condition."""
        # Check element type
        if "element_type" in condition and element.element_type != condition["element_type"]:
            return False

        # Check attributes
        if "attributes" in condition:
            for attr_name, expected_values in condition["attributes"].items():
                if attr_name not in element.attributes:
                    return False

                attr_value = element.attributes[attr_name]

                if isinstance(expected_values, list):
                    if attr_value not in expected_values:
                        return False
                elif expected_values == "not_none":
                    if not attr_value:
                        return False
                elif expected_values == "search_pattern":
                    # Check if placeholder or aria-label contains search-related text
                    search_indicators = ["search", "find", "query", "lookup"]
                    if not any(indicator in attr_value.lower() for indicator in search_indicators):
                        return False
                elif expected_values == "nav_pattern":
                    # Check if aria-label contains navigation-related text
                    nav_indicators = ["nav", "menu", "navigation", "breadcrumb"]
                    if not any(indicator in attr_value.lower() for indicator in nav_indicators):
                        return False

        return True

    def _create_component(
        self, element: Element, rule: GroupingRule, all_elements: list[Element]
    ) -> FunctionalComponent:
        """Create a functional component from an element and rule."""
        component_id = f"{rule.component_type.value}_{element.element_id}"

        # Extract attributes based on rule
        attributes = []
        if rule.attribute_extractors:
            for extractor_config in rule.attribute_extractors:
                attr_name = extractor_config["name"]
                extractor_type = extractor_config["extractor"]

                if extractor_type == "text_content":
                    value = element.text_content
                elif extractor_type == "input_type":
                    value = element.attributes.get("type", "")
                elif extractor_type == "is_disabled":
                    value = "disabled" in element.attributes
                elif extractor_type == "placeholder":
                    value = element.attributes.get("placeholder", "")
                elif extractor_type == "is_required":
                    value = "required" in element.attributes
                elif extractor_type == "current_value":
                    value = element.attributes.get("value", "")
                elif extractor_type == "dropdown_options":
                    value = self._extract_dropdown_options(element)
                elif extractor_type == "selected_option":
                    value = self._extract_selected_option(element)
                elif extractor_type == "is_multiple":
                    value = "multiple" in element.attributes
                elif extractor_type == "adjacent_button":
                    value = self._find_adjacent_button(element, all_elements)
                elif extractor_type == "nav_links":
                    value = self._extract_nav_links(element, all_elements)
                elif extractor_type == "current_nav_item":
                    value = self._extract_current_nav_item(element)
                elif extractor_type == "form_action":
                    value = element.attributes.get("action", "")
                elif extractor_type == "form_method":
                    value = element.attributes.get("method", "get")
                elif extractor_type == "form_fields":
                    value = self._extract_form_fields(element, all_elements)
                else:
                    value = None

                if value is not None:
                    attributes.append(
                        ComponentAttribute(
                            name=attr_name, value=value, description=f"Extracted from {extractor_type}"
                        )
                    )

        return FunctionalComponent(
            component_id=component_id,
            component_type=rule.component_type,
            elements=[element],
            attributes=attributes,
            is_interactive=element.is_interactive,
            text_content=element.text_content,
            selector=element.selector,
            bounding_box=element.bounding_box,
        )

    def _extract_dropdown_options(self, element: Element) -> list[str]:
        """Extract options from a select element."""
        # This would need to be implemented with actual DOM access
        # For now, return a placeholder
        return []

    def _extract_selected_option(self, element: Element) -> str:
        """Extract the selected option from a select element."""
        return element.attributes.get("value", "")

    def _find_adjacent_button(self, element: Element, all_elements: list[Element]) -> str:
        """Find a button adjacent to the given element."""
        if not element.bounding_box:
            return ""

        # Look for buttons near this element
        for other_element in all_elements:
            if (
                other_element.element_type == "button"
                and other_element.bounding_box
                and self._are_elements_adjacent(element, other_element)
            ):
                return other_element.text_content
        return ""

    def _extract_nav_links(self, element: Element, all_elements: list[Element]) -> list[str]:
        """Extract navigation links from a nav element."""
        links = []
        for child_id in element.children_ids:
            for other_element in all_elements:
                if other_element.element_id == child_id and other_element.element_type == "a":
                    links.append(other_element.text_content)
        return links

    def _extract_current_nav_item(self, element: Element) -> str:
        """Extract the current navigation item."""
        # Look for aria-current or similar indicators
        return element.attributes.get("aria-current", "")

    def _extract_form_fields(self, element: Element, all_elements: list[Element]) -> list[str]:
        """Extract form fields from a form element."""
        fields = []
        for child_id in element.children_ids:
            for other_element in all_elements:
                if other_element.element_id == child_id and other_element.is_interactive:
                    fields.append(other_element.element_type)
        return fields

    def _are_elements_adjacent(self, element1: Element, element2: Element) -> bool:
        """Check if two elements are adjacent based on their bounding boxes."""
        if not element1.bounding_box or not element2.bounding_box:
            return False

        box1 = element1.bounding_box
        box2 = element2.bounding_box

        # Check if elements are close to each other (within 20 pixels)
        distance_threshold = 20

        # Calculate center points
        center1_x = box1["x"] + box1["width"] // 2
        center1_y = box1["y"] + box1["height"] // 2
        center2_x = box2["x"] + box2["width"] // 2
        center2_y = box2["y"] + box2["height"] // 2

        # Calculate distance
        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

        return distance <= distance_threshold

    def _group_input_labels(self, elements: list[Element]) -> list[FunctionalComponent]:
        """Group input elements with their associated labels."""
        components = []
        processed_elements = set()

        for element in elements:
            if element.element_id in processed_elements:
                continue

            if element.element_type in ["input", "textarea", "select"]:
                # Look for associated label
                label_element = self._find_associated_label(element, elements)

                if label_element:
                    component_elements = [element, label_element]
                    processed_elements.add(element.element_id)
                    processed_elements.add(label_element.element_id)
                else:
                    component_elements = [element]
                    processed_elements.add(element.element_id)

                # Create component
                component = FunctionalComponent(
                    component_id=f"input_group_{element.element_id}",
                    component_type=ComponentType.INPUT_FIELD,
                    elements=component_elements,
                    attributes=[
                        ComponentAttribute("placeholder", element.attributes.get("placeholder", "")),
                        ComponentAttribute("type", element.attributes.get("type", "text")),
                        ComponentAttribute("required", "required" in element.attributes),
                        ComponentAttribute("label", label_element.text_content if label_element else ""),
                    ],
                    is_interactive=element.is_interactive,
                    text_content=element.text_content,
                    selector=element.selector,
                    bounding_box=element.bounding_box,
                )
                components.append(component)

        return components

    def _find_associated_label(self, input_element: Element, all_elements: list[Element]) -> Element | None:
        """Find the label associated with an input element."""
        # Check for label with 'for' attribute matching input's id
        input_id = input_element.attributes.get("id", "")
        if input_id:
            for element in all_elements:
                if element.element_type == "label" and element.attributes.get("for") == input_id:
                    return element

        # Check for label that contains the input
        for element in all_elements:
            if element.element_type == "label" and input_element.element_id in element.children_ids:
                return element

        # Check for nearby label elements
        if input_element.bounding_box:
            for element in all_elements:
                if (
                    element.element_type == "label"
                    and element.bounding_box
                    and self._are_elements_adjacent(input_element, element)
                ):
                    return element

        return None

    def _group_buttons(self, elements: list[Element]) -> list[FunctionalComponent]:
        """Group related buttons together."""
        components = []
        processed_elements = set()

        # Look for button groups (buttons that are close together)
        button_elements = [
            e for e in elements if e.element_type == "button" and e.element_id not in processed_elements
        ]

        for button in button_elements:
            if button.element_id in processed_elements:
                continue

            # Find nearby buttons
            nearby_buttons = [button]
            processed_elements.add(button.element_id)

            for other_button in button_elements:
                if (
                    other_button.element_id not in processed_elements
                    and button.bounding_box
                    and other_button.bounding_box
                    and self._are_elements_adjacent(button, other_button)
                ):
                    nearby_buttons.append(other_button)
                    processed_elements.add(other_button.element_id)

            if len(nearby_buttons) > 1:
                # Create button group component
                component = FunctionalComponent(
                    component_id=f"button_group_{button.element_id}",
                    component_type=ComponentType.BUTTON,
                    elements=nearby_buttons,
                    attributes=[
                        ComponentAttribute("button_count", len(nearby_buttons)),
                        ComponentAttribute("button_texts", [b.text_content for b in nearby_buttons]),
                    ],
                    is_interactive=True,
                    text_content=", ".join(b.text_content for b in nearby_buttons),
                    selector=button.selector,
                    bounding_box=button.bounding_box,
                )
                components.append(component)

        return components

    def _group_cards(self, elements: list[Element]) -> list[FunctionalComponent]:
        """Group elements that form card-like structures."""
        components = []
        processed_elements = set()

        # Look for card containers (divs with card-like styling or structure)
        for element in elements:
            if element.element_id in processed_elements:
                continue

            if element.element_type == "div" and self._looks_like_card(element, elements):
                # Find all child elements that belong to this card
                card_elements = self._find_card_elements(element, elements)

                if len(card_elements) > 1:
                    processed_elements.update(e.element_id for e in card_elements)

                    component = FunctionalComponent(
                        component_id=f"card_{element.element_id}",
                        component_type=ComponentType.CARD,
                        elements=card_elements,
                        attributes=[
                            ComponentAttribute("element_count", len(card_elements)),
                            ComponentAttribute(
                                "has_header",
                                any(
                                    e.element_type in ["h1", "h2", "h3", "h4", "h5", "h6"]
                                    for e in card_elements
                                ),
                            ),
                            ComponentAttribute(
                                "has_buttons", any(e.element_type == "button" for e in card_elements)
                            ),
                        ],
                        is_interactive=any(e.is_interactive for e in card_elements),
                        text_content=element.text_content,
                        selector=element.selector,
                        bounding_box=element.bounding_box,
                    )
                    components.append(component)

        return components

    def _looks_like_card(self, element: Element, all_elements: list[Element]) -> bool:
        """Check if an element looks like a card container."""
        # Check for card-like class names
        class_name = element.attributes.get("class", "").lower()
        card_indicators = ["card", "panel", "box", "container", "widget"]

        if any(indicator in class_name for indicator in card_indicators):
            return True

        # Check if it has multiple child elements including headers and content
        if len(element.children_ids) >= 3:
            return True

        return False

    def _find_card_elements(self, card_element: Element, all_elements: list[Element]) -> list[Element]:
        """Find all elements that belong to a card."""
        card_elements = [card_element]

        # Add direct children
        for child_id in card_element.children_ids:
            for element in all_elements:
                if element.element_id == child_id:
                    card_elements.append(element)
                    break

        return card_elements
