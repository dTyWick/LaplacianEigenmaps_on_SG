# LaplacianEigenmaps_on_SG
    
<p align="center">
  <img src="images/HanoiFlower_n8_p5.png" width="45%" />
  <img src="images/R8_SG_Analysis.png" width="45%" /> 
</p>

Through numerical investigations of the resistance metric on Poisson-weighted graph approximations of the Sierpinski Gasket (SG), we provide evidence of the extension of the Laplacian Eigenmaps technique to non-smooth spaces, using SG as a foundational example.

This project is through the [UCONN Fractals REU Program](https://mathreu.uconn.edu/projects/fractals/). This work would not have been possible without the help from collaborators Ben Roark and Ranjeev Iyer and mentors Bernard Akwei, Luke Rogers, and Alexander Teplyaev.


## Background Information

* First, see both the [abstract](/backgroundInfo/JMM_Abstract.pdf) we submitted (and which was accepted) to JMM 2026 and the [comprehensive presentation](/backgroundInfo/Comprehensive_Presentation.pdf) which we gave at the end of the 2025 UCONN REU for the context of this work.

* NOTE: This work is ongoing, so some of the work in the previous abstract and presentation may be wrong or out of date. That being said, they provide good context for the below work.

## Reproducibility

To replicate the results in this repository, please follow the steps below to set up the environment and run the analysis. **NOTE**: The venv was created using Python 3.13.7.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dTyWick/LaplacianEigenmaps_on_SG.git
   cd LaplacianEigenmaps_on_SG
   ```

2. **Create and activate the virtual environment (named `randSG`):**
   * **Windows:**
     ```bash
     python -m venv randSG
     randSG\Scripts\activate
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv randSG
     source randSG/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r reqs.txt
   ```

4. **Run the code:**
   Launch the Jupyter environment to view the notebooks.
   ```bash
   jupyter lab
   ```

## Project Structure

The project has the following structure:

```text
├── backgroundInfo                          # Abstract and Presentation Describing Work
│   ├── Comprehensive_Presentation.pdf      
│   └── JMM_Abstract.pdf
├── images                                  # images generated from notebooks
├── notebooks                               # jupyter notebooks containing codes
|   ├── construct_and_plot_graphs.ipynb     # creation/visualization of random hanoi graphs
|   ├── image_generation.ipynb              # code for generating most images found in images folder
|   ├── var_exploration.ipynb               # analysis of the variance and expectation of effective resistance on H_n
|   ├── eigenmaps.ipynb                     # create/explore the convergence of eigenmaps on random/deterministic H_n
├── fractal_functions.py                    # python file containing relevant functions. Used extensively in notebooks
├── LICENSE                                 # GPL 3.0
├── README.md                               # This file
└── reqs.txt                                # library requirements for venv
```

