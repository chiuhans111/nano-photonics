def mostmvwrite(filename: str, var_names: list, *var_values):
    """
    寫入 MOST multi variable file

    Parameters
    ----------
    filename: str
        檔案名稱(路徑)

    var_name: list[str]
        變數名稱(依序)，例如 ["A", "B", "C"]

    var_values: list[array]
        要寫入的變數內容
    """

    assert len(var_names) == len(var_values), \
        f"變數名稱應該與所提供的值相同, {len(var_names)} vs {len(var_values)}"

    # the beginning of the content
    multi_var_content = "!MOSTMultiVarFormat1"

    # write all variable name
    multi_var_content += "\n"+" ".join(var_names)

    # loop through all variable value
    for i in range(len(var_values[0])):
        multi_var_content += "\n"+" ".join([str(v[i]) for v in var_values])

    # open file ane write content
    with open(filename, 'w') as f:
        f.write(multi_var_content)
