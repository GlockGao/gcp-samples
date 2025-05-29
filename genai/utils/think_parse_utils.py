def parse_response_with_tags(
    response_content: str, tag_name: str
) -> tuple[str | None, str]:
    start_tag = f"<{tag_name}>"
    end_tag = f"{tag_name}>"

    start_index = response_content.find(start_tag)
    end_index = -1
    if start_index != -1:
        end_index = response_content.find(end_tag, start_index + len(start_tag))

    tagged_content = None
    main_content = response_content.strip()

    if start_index != -1 and end_index != -1:
        tagged_content = response_content[
            start_index + len(start_tag) : end_index
        ].strip()
        part_before_tag = response_content[:start_index].strip()
        part_after_tag = response_content[end_index + len(end_tag) :].strip()

        if part_before_tag and part_after_tag:
            main_content = f"{part_before_tag}\n\n{part_after_tag}"
        elif part_before_tag:
            main_content = part_before_tag
        elif part_after_tag:
            main_content = part_after_tag
        else:
            main_content = ""

    elif start_index != -1 and end_index == -1:
        pass

    return tagged_content, main_content