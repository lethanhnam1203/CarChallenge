
# Train Neural Networks
## Setup
- Create a virtual environment and install the required packages with `pip install -r requirements.txt`
- Store all image files of the dataset inside `imgs` folder
- Store the label csv file of the dataset inside `labels` folder
## Execution
Run the following bash script from root directory. This execution will train and evaluate four different models on the dataset.

`
./execute.sh 
`

Choose the model with the best **validation** result for the image prediction flask app. In my case, it was `"resnet18"`.


# Model Inference with a Flask App
## Setup
Make sure that you have Docker installed e.g. Docker Desktop.
## Execution
- Run `docker compose up` from the root directory.
- Move to the `test` folder with `cd test` and then run `python3 test.py` there. This execution will carry out inference on the sampled test images stored inside the `test_sampled_imgs` subfolder. In the terminal, predictions from the model as well as ground truths will be printed

