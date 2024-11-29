import numpy as np

from typing import Tuple, List
import numpy.typing as npt

def fields_to_rows(fields: Tuple[npt.NDArray, ...]) -> Tuple[npt.NDArray, List[Tuple]]:
    shapes = []
    rows_agg = np.empty(shape=(fields[0].shape[0],0))
    for field in fields:
        rows = field.reshape(field.shape[0], -1)
        rows_agg = np.concatenate((rows_agg, rows), axis=1)
        shapes.append(field.shape)
        
    return rows_agg, shapes

def rows_to_fields(rows: npt.NDArray, shapes) -> Tuple[npt.NDArray, ...]:
    num_fields = len(shapes)
    
    start_idx = 0
    fields = []
    for i in range(num_fields):
        row_field_len = 1
        for j in range(1, len(shapes[i])):
            row_field_len *= shapes[i][j]
        row_field = np.array(rows[:, start_idx:start_idx+row_field_len])
        start_idx += row_field_len
        
        field = row_field.reshape(shapes[i])
        fields.append(field)
        
    return tuple(fields)