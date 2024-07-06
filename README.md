
# Set up
Create a virtual environment and install the required packages with `pip install -r requirements.txt`

# Train neural networks
Run the following bash script from root directory. This execution will train and evaluate four different models on the dataset.

`
./execute.sh 
`

Choose the model with the best **validation** result for the image prediction flask app. In my case, it was `"resnet18"`.


# Run the flask app
Make sure that you have Docker installed e.g. Docker Desktop. Just run `docker compose up` from the root directory.

# Try model inference via the flask app
Move to the `test` folder with `cd test` and then run `python3 test.py` there. This execution will carry out inference on the sampled test images stored inside the `test_sampled_imgs` subfolder. In the terminal, predictions from the model as well as ground truths will be printed

