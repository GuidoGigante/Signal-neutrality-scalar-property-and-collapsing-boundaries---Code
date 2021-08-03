# Signal-neutrality-scalar-property-and-collapsing-boundaries---Code
Python code to replicate the results of the paper "Signal neutrality, scalar property, and collapsing boundaries as consequences of a learned multi-time scale strategy".

The only .py file includes all the needed functions (apart from quite standard Python imports, like numpy, pandas, scipy.special, and time). It also includes a main function that trains an agent and uses it to generate a number of episodes to compute relevant statistics (like performance, average response times, signal neutrality and scalar property metrics, etc.) to illustrate the use of the different functions.

The code is optimized through Numba. If you don't want to use Numba, just comment out "from numba import jit", "import numba.typed as numba_type" and all the "@jit(nopython=True)" before the function definitions; then substitute the lines:
    history_episodes = numba_type.List()
    w_history = numba_type.List()
    th_history = numba_type.List()
with:
    history_episodes = []
    w_history = []
    th_history = []
    
and the code should run as is (but slower; I don't know exactly how much slower, maybe a factor 5).
