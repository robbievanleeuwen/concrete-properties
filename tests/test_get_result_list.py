import pytest
import time
import numpy as np
from numpy import random, ndarray as NdArray, testing
from dataclasses import dataclass, field
from colorama import Fore, Style


@dataclass(order=True)
class UltimateBendingResults:
    # bending angle
    # theta: float

    # ultimate neutral axis depth
    # d_n: float = 0
    # k_u: float = 0

    # resultant actions
    n: float = 0
    m_x: float = 0
    m_y: float = 0
    m_xy: float = 0

    # label
    # label: str | None = field(default=None, compare=False)


@dataclass
class BiaxialBendingResults:
    n: float
    results: list[UltimateBendingResults] = field(default_factory=list)

    def get_results_lists(self) -> tuple[list[float], list[float]]:
        # build list of results
        m_x_list = []
        m_y_list = []

        for result in self.results:
            m_x_list.append(result.m_x)
            m_y_list.append(result.m_y)

        return m_x_list, m_y_list


@dataclass
class NewBiaxialBendingResults:
    n: float
    # results: list[UltimateBendingResults] = field(default_factory=list)
    results: NdArray  # swap the list out for an NdArray

    @classmethod
    def from_list(
        cls,
        n: float,
        results: list[UltimateBendingResults]
        ):

        dtype = [('m_x', float), ('m_y', float)]
        # ... specifies the dtype for both m_x/y
        
        np_results: NdArray = np.array(
            [
            (result_.m_x, result_.m_y) 
            for result_ in results
            ], dtype=dtype )
        
        return cls(n, np_results)

    def get_results_lists(self) -> tuple[np.ndarray, np.ndarray]:
        return self.results['m_x'], self.results['m_y']


def generate_test_data(num_results: int) -> list[UltimateBendingResults]:
    return [UltimateBendingResults(
        m_x=random.rand(), m_y=random.rand()
        # examined operation is extracting the m_x/y from the results, only this is needed
        ) for _ in range(num_results)]

# --- ⏲️
def time_execution(func, *args, **kwargs) -> tuple:
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()

    return (result, end_time - start_time)

# --- 🧪
@pytest.mark.parametrize("num_results", [10, 100, 1_000, 10_000, 100_000, 1_000_000])
def test_get_results_lists(num_results) -> None:
    random.seed(42)  # seed set for reproducible test outcome
    
    test_data: list[UltimateBendingResults] = generate_test_data(num_results)
    
    # --- Extract m_x/y from the same random test_data
    original_results = BiaxialBendingResults(n=1_000, results=test_data) # same n_value for both, unlike inrweaxrion
    new_results = NewBiaxialBendingResults.from_list(n=1_000, results=test_data)
    
    # --- Time the execution of both operations
    (original_m_x, original_m_y), original_time = time_execution(original_results.get_results_lists) # Original
    (new_m_x, new_m_y), new_time = time_execution(new_results.get_results_lists)  # New cl
    
    # --- Check correctness between operations
    assert len(original_m_x) == len(new_m_x) == num_results  # check no. of results
    assert len(original_m_y) == len(new_m_y) == num_results
    
    testing.assert_allclose(original_m_x, new_m_x) # check same values of m_x/y 
    testing.assert_allclose(original_m_y, new_m_y)
    
    # --- Calc. speedup 
    speedup = original_time / new_time if new_time > 0 else float('inf')
    assert new_time < original_time # 

    # --- 📝 Compare
    print(f"\nFor {num_results:,} results:")
    
    # print(f"🐌 Original method: {original_time:.6f} seconds")
    # print(f"🐎 New method: {new_time:.6f} seconds")
    print(f"🐌 Original method: {original_time:.3e} seconds]")
    print(f"🐎 New method: {new_time:.3e} secons]")
    print(f"⚡ SPEEDUP: {speedup:.2f}x")

if __name__ == "__main__":
    pytest.main([__file__])
    test_get_results_lists(num_results=555_555)
