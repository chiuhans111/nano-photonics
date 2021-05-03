# 讀取 most 相關檔案

import os
import numpy as np


def parse_split(s: str, sep=' ', T: type = str):
    """
    將字串切割，去掉不必要頭尾，轉換成指定類型

    Parameters
    ----------
    s: str
        要切割的字串
    sep: str
        切割符號
    T: type
        轉換成類型
    """
    arr = []
    for v in s.strip().split(sep):
        arr.append(T(v.strip()))
    return arr


def get_variable(filename):
    """
    get variable values from a MOST multi variable file

    Parameters
    ----------
    filename: str
        MOST multi variable file 檔案路徑


    Returns
    -------
    table[key] = value
        key: str
            變數名稱
        value: ndarray
            變數數值陣列
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        var_names = parse_split(lines[1])

        values = []
        for line in lines[2:]:
            values.append(parse_split(line, ' ', float))

        values = np.array(values)
        table = {}
        for i, v in enumerate(var_names):
            table[v] = values[:, i]
    return table


def get_result(path):
    """
    取得並閱讀 MOST 掃描結果 .dat

    Parameters
    ----------
    path: str
        MOST 優化輸出 檔案路徑 .../XXXX_work


    Returns
    -------
    table[key] = value
        key: str
            輸出結果名稱
        value: ndarray
            變數數值陣列
    """
    # get result from xxx_work/result
    result_folder = os.path.join(path, "results")
    results = {}
    for file in os.listdir(result_folder):
        filename, ext = os.path.splitext(file)
        if ext == ".dat":
            result = []
            with open(os.path.join(result_folder, file), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    result.append(parse_split(line, ' ', float))

            results[filename] = np.array(result)[:, 1:]
    return results


def get_result_vtk(folder):
    """
    取得並閱讀 MOST 掃描結果 .vtk

    Parameters
    ----------
    path: str
        MOST 優化輸出 檔案路徑 .../XXXX_work


    Returns
    -------
    table[key] = value
        key: str
            輸出結果名稱
        value: ndarray
            變數數值陣列
    """
    result_folder = os.path.join(folder, "results")

    results = {}
    for file in os.listdir(result_folder):
        filename, ext = os.path.splitext(file)
        if ext == ".vtk":
            with open(os.path.join(result_folder, file), 'r') as f:
                lines = f.readlines().__iter__()

                for line in lines:
                    # 跳過前面所有資訊直到讀取到 DIMENSION 開頭
                    if line.startswith("DIMENSIONS"):
                        dimension_values = line.strip().split(' ')[1:]
                        dimension = [int(v) for v in dimension_values]
                        break

                for line in lines:
                    # 跳過中間所有直到 LOOKUP_TABLE 出現
                    if line.startswith("LOOKUP_TABLE"):
                        break

                values = []
                for line in lines:
                    # 讀取後面所有資訊
                    content = line.strip()
                    if len(content) == 0:
                        continue
                    values.append(float(content))

                results[filename] = np.reshape(
                    np.array(values), dimension, "F")
    return results
