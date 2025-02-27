# BSc Thesis - Energy Consumption Forecasting

##

## Folder Structure

- **IDEAL/** - Contains the **IDEAL Household Energy Dataset**, which was used for this project. Credit:\
  IDEAL Household Energy Dataset [dataset]. University of Edinburgh. School of Informatics. [https://doi.org/10.7488/ds/2836](https://doi.org/10.7488/ds/2836)
- **LSTM/** - Contains:
  - The dataset used to train the LSTM model (**dataset/** folder).
  - Python scripts for training and saving the LSTM model.
- **TFT/** - Contains:
  - The dataset used to train the TFT model (**dataset/** folder).
  - Python scripts for training and saving the TFT model.
- **Models/** - Stores the trained **Keras** models.
- **requirements.txt** - Lists the Python dependencies required to run this project.
- **.vscode/** - (Optional) Contains VS Code workspace settings.

## Setup and Installation

### Virtual Environment

The `venv/` folder is a **virtual environment**, which helps manage dependencies for this project. It is recommended to use it to avoid package conflicts. Best practices suggest:

- **Not uploading ****`venv/`** to the repository, as it can be recreated using `requirements.txt`.
- Instead, install dependencies using:
  ```bash
  python -m venv venv  # Create virtual environment
  source venv/bin/activate  # On macOS/Linux
  venv\Scripts\activate  # On Windows
  pip install -r requirements.txt  # Install dependencies
  ```


## Usage

After setting up the environment, you can train models using the scripts in the `LSTM/` and `TFT/` folders. Models are saved in the `Models/` directory.

## Citation

If you use the **IDEAL Household Energy Dataset**, please credit:

> IDEAL Household Energy Dataset [dataset]. University of Edinburgh. School of Informatics. [https://doi.org/10.7488/ds/2836](https://doi.org/10.7488/ds/2836)





