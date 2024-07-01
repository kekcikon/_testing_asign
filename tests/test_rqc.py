#!/usr/bin/env python3
import pytest
import numpy as np
import ctypes
from tests import SRC, compile

@pytest.fixture
def libfact():
    source = SRC / "rqc.cpp"
    compile(source)
    yield ctypes.CDLL(str(source.with_suffix(".so")))

@pytest.mark.parametrize("x, y, res", [(np.eye(3, dtype=np.double), np.linspace(1, 9, 9).reshape((3,3)), np.linspace(1, 9, 9).reshape((3,3))),
                                       (np.linspace(1, 9, 9).reshape((3,3)), np.eye(3, dtype=np.double), np.linspace(1, 9, 9).reshape((3,3))),
                                        (np.zeros((3,3), dtype=np.double), np.random.random(9).reshape((3,3)), np.zeros((3,3), dtype=np.double)), 
                                        (np.random.random(9).reshape((3,3)), np.zeros((3,3), dtype=np.double), np.zeros((3,3), dtype=np.double)),
                                        (np.linspace(1, 9, 9).reshape((3,3)), np.linspace(1, 9, 9).reshape((3,3)), np.linspace(1, 9, 9).reshape((3,3))@np.linspace(1, 9, 9).reshape((3,3)))
                                        ])
def test_square(libfact, x, y, res):
    libfact.mulMat.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_double)
    ]
    answer = np.zeros((x.shape[0], y.shape[1]), dtype=np.double)
    x_c = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    answer_c = answer.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    libfact.mulMat(x_c, x.shape[0], x.shape[1], y_c, y.shape[0], y.shape[1], answer_c)     
    assert (answer == res).all()

@pytest.mark.parametrize("x, y, res", [(np.linspace(1, 6, 6).reshape((2,3)), np.eye(3), np.linspace(1, 6, 6).reshape((2,3))),
                                        (np.linspace(1, 6, 6).reshape((2,3)), np.zeros((3,3)), np.zeros((2,3)))
                                        ])
def test_rect(libfact, x, y, res):
    answer = np.zeros((x.shape[0], y.shape[1]), dtype=np.double)
    x_c = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    answer_c = answer.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    libfact.mulMat(x_c, x.shape[0], x.shape[1], y_c, y.shape[0], y.shape[1], answer_c)     
    assert (answer == res).all()

@pytest.mark.parametrize("x, y, res", [(np.array([[2, 5, 7], [6,3,4], [5,-2,-3]], dtype=np.double), np.linalg.inv(np.array([[2, 5, 7], [6,3,4], [5,-2,-3]])), np.eye(3)),
                                       (np.linalg.inv(np.array([[2, 5, 7], [6,3,4], [5,-2,-3]])), np.array([[2, 5, 7], [6,3,4], [5,-2,-3]], dtype=np.double), np.eye(3))

                                        ])
def test_inv_matrix(libfact, x, y, res):
    answer = np.zeros((x.shape[0], y.shape[1]), dtype=np.double)
    x_c = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_c = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    answer_c = answer.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    libfact.mulMat(x_c, x.shape[0], x.shape[1], y_c, y.shape[0], y.shape[1], answer_c)     
    assert np.allclose(answer, res)