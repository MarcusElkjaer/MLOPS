# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M13)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [y] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [y] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [y] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x] Create a FastAPI application that can do inference using your model (M22)
* [y] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [x] Write API tests for your application and setup continues integration for these (M24)
* [x] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [x] Check how robust your model is towards data drifting (M27)
* [x] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [x] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [x] Write some documentation for your application (M32)
* [x] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

## Group information


### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer: 34


--- question 1 fill here ---


### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:
 S224743, s224803, s214625, s224751, s214636  


--- question 2 fill here ---


### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:


In our project, we used the Transformers library from Hugging Face to work with pre-trained language models. We used AutoTokenizer, which helps convert text into numbers (tokens) that the model can understand. Instead of manually figuring out which tokenizer to use, AutoTokenizer automatically picks the right one based on the model we choose.

We also used AutoModel, which makes it easy to load the correct deep learning model without needing to know all the technical details. By just providing the model’s name or path, AutoModel takes care of setting up everything for us. This made our work faster and easier, allowing us to focus more on building our project.


--- question 3 fill here ---




## Coding environment


> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.


### Question 4


> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:
We managed dependencies by maintaining a requirements.txt file for production and testing needs furthermore for containerized workloads, we used Dockerfiles that installed the dependencies from these requirement files, ensuring that the build was both repeatable and isolated from local machine variations.
A new team member wanting to replicate our environment would simply clone the repository, create a virtual environment (e.g., via python -m venv .venv), activate it, and run pip install -r requirements.txt. This setup guarantees they are using the exact same libraries and versions as everyone else. If Docker is preferred, they would just build the Docker image from our provided Dockerfile, which installs and locks the dependencies automatically.

--- question 4 fill here ---


### Question 5


> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We've mostly followed the default structure of the Cookiecutter template but added a few things to improve organization. One key addition is a dedicated folder for our frontend API, which keeps all related code in one place and makes it easier to manage and also more organized. We also introduced a "workflow" folder to store our GitHub workflow files, ensuring our automation scripts are neatly organized. In addition, we created standalone files for utility functions to keep things modular and avoid clutter. These improvements help maintain a clean project structure, making it easier to work on and scale over time.

--- question 5 fill here ---

### Question 6
> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

To maintain code quality and consistency, we used several tools and practices throughout the project. For formatting, we relied on Black, ensuring that our code adhered to a consistent style. Ruff was used as a linter to enforce PEP 8 compliance and identify potential issues. These tools were automated using pre-commit hooks defined in .pre-commit-config.yaml, ensuring quality checks were run before committing code.
Documentation was another area of focus. Docstrings were added to critical functions and classes, explaining their purpose, inputs, and outputs. This documentation is essential in larger projects, where the codebase grows complex. Good documentation, combined with type hints, makes it easier for new team members to understand and contribute to the project.
By following these practices, we were sure that the code is not only readable and consistent but also maintainable as the project evolves.
We also made a github page for our documentation that can be found here: https://marcuselkjaer.github.io/MLOPS/

--- question 6 fill here ---
## Version control
> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.


### Question 7


> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total, we implemented 26 tests to ensure the reliability and functionality of our codebase. We began by focusing on testing the tasks module, which allowed us to verify that core functionalities were working as expected while familiarizing ourselves with the testing framework.
Once we gained confidence, we proceeded to implement tests for the model, training pipeline, and data processing components. In these tests, we utilized mocking techniques to simulate various scenarios and dependencies, such as file handling and API calls. This approach significantly improved testing efficiency by reducing reliance on external dependencies and speeding up execution time.
By systematically covering different aspects of the project, our tests now provide robust validation for key functionalities, helping to catch potential issues early and ensuring the stability of our system across different scenarios.

--- question 7 fill here ---


### Question 8


> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our project is 70%, as reported by the GitHub Actions workflow. While this level of coverage ensures that a significant portion of the application has been tested, it also indicates that certain areas, such as data.py (42% coverage), could benefit from additional tests to improve confidence in their reliability.
Even if our code had 100% coverage, it would not guarantee that it is completely error-free. Code coverage only measures the lines of code executed during tests; it does not validate whether all potential edge cases, integration points, or unintended behaviors have been addressed. For example, high coverage may miss logical errors, incorrect assumptions, or untested scenarios in edge cases.

--- question 8 fill here ---


### Question 9


> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

In our project, we only used the main branch for development and did not incorporate branches or pull requests into our workflow. While this simplified our development process, it also meant that all changes were directly made to the main branch, potentially increasing the risk of introducing bugs or conflicts without prior review.
Branches and pull requests are important tools in version control for several reasons. Branches programmers to work on separate features, bug fixes, or experiments without affecting the main codebase. This isolation prevents incomplete or experimental code from disrupting the functionality of the application.
Pull requests provide an opportunity for code review before merging changes into the main branch. This process ensures that the code meets quality standards, follows best practices, and integrates smoothly with existing functionality. It also facilitates collaboration by enabling programmers to discuss changes, suggest improvements, and ensure consensus before any code is merged.

--- question 9 fill here ---


### Question 10


> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

Yes, we used DVC (Data Version Control) for managing data in our project. The DVC configuration was linked to a Google Cloud Storage bucket (gs://mlops_data_reddit_bucket/), as specified in the configuration file. By versioning our data with DVC, we ensured that the raw, preprocessed, and final datasets were consistently tracked and could be reproduced reliably across different environments.
It also made sharing data within our team much simpler and eliminated concerns about GitHub’s file size limits, as large datasets were stored in the cloud while version tracking remained within our codebase. This improved collaboration, workflow efficiency and overall a nicer work process.

--- question 10 fill here ---

### Question 11


> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>


> Answer:

Our continuous integration system relies on GitHub Actions and incorporates several workflows designed to streamline different stages of development. The unit testing workflow ensures compatibility across platforms by running tests on both ubuntu-latest and windows-latest environments. It uses Python 3.11 for consistency and includes ruff for linting to maintain high code quality. Tests are executed using pytest, with a code coverage report generated via coverage. To enhance efficiency, the workflow utilizes dependency caching with the actions/cache feature, reducing setup time for subsequent runs.
Another workflow handles the creation and deployment of Docker images. This process involves building the application image from a specified Dockerfile, uploading it to Google Artifact Registry, and deploying it to Google Cloud Run. The deployed service is configured for public access and allocated 4GB of memory to ensure reliable performance. Additionally, untagged images in the registry are periodically removed to optimize storage usage.
Another workflow tracks changes to data files. Whenever modifications occur, it automatically calculates updated dataset statistics and saves the results as artifacts. This guarantees that any alterations to the data are processed and logged promptly, ensuring the project remains well-documented and up-to-date.

--- question 11 fill here ---


## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12


> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>


> Answer:

We configured our experiments using a combination of YAML configuration files and Python scripts. The configuration file, config.yaml, defines key hyperparameters and settings for training and optimization. This includes values like the batch size, learning rate, L2 regularization, number of epochs, and the dataset path. 
Our train.py integrates with the Hydra framework, which automatically loads the configuration file. The main function initializes and manages the training process, including hyperparameter optimization with Optuna.

--- question 12 fill here ---


### Question 13


> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>

> Answer:

We used a YAML configuration file (config.yaml) managed by the Hydra framework. This centralized configuration defined all key parameters, such as the learning rate, batch size, training split, and number of epochs. Hydra dynamically loads this configuration during execution, making sure that each experiment uses a consistent set of parameters. 
To guarantee reproducibility, we controlled random seeds across all relevant components. Using the seed_randoms function in train.py
Additionally, after training, models and their configurations were saved, allowing us to reproduce results easily by reloading the saved models. This approach streamlined experimentation and made it easier to track and compare different runs.

--- question 13 fill here ---


### Question 14


> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

Performing a hyperparameter sweep in optuna and logging the results to WANDB, we can log the experiments that we performed.

From the parallel coordinates of the hyperparameter sweep, we can observe that a lower batch size and l2 penalty, and a learning rate of around 2e-5 tends to yield the best results. The best trial had a validation loss of around 0.0515.

Plotting the validation loss over our epochs, we see that the models tend to fit quite quickly, likely as a result of using BERT as a base model and training from there. 

--- question 14 fill here ---

Performing a hyperparameter sweep in optuna and logging the results to WANDB, we can log the experiments that we performed.

![this figure](figures/Parrelelplot.png)


From the parallel coordinates of the hyperparameter sweep, we can observe that a lower batch size and l2 penalty, and a learning rate of around 2e-5 tends to yield the best results. The best trial had a validation loss of around 0.0515.

![this figure](figures/val_loss.png)

Plotting the validation loss over our epochs, we see that the models tend to fit quite quickly, likely as a result of using BERT as a base model and training from there. 


### Question 15


> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:
We used Docker to containerize our components in our project ensuring a reproducible environment. We have 2 dockerfiles. One for the API, dockerfiles/api.Dockerfile and one for training, dockerfiles/train.Dockerfile.
The API Dockerfile (dockerfiles/api.Dockerfile) defines the setup for serving our FastAPI application. It includes steps to install dependencies, copy source code, and run the backend service alongside a frontend application. For the train we would run:
docker build -f dockerfiles/train.dockerfile -t sentiment-train:latest .
docker push gcr.io/mlops-448109/sentiment-train:latest
and then run:
docker run -it --rm gcr.io/mlops-448109/sentiment-train:latest



--- question 15 fill here ---


### Question 16


> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:
When debugging our project, we primarily used Python's logging module to track the execution flow and identify issues. Logging was used in scripts like data.py and train.py to monitor key steps, such as loading data, preprocessing, and training. by using logging we were able to quickly pinpoint problems, such as file path issues or data inconsistencies, without adding unnecessary spam in the output.


While we did not focus that much on profiling , we made some attempts to use cProfile along the way to understand potential bottlenecks in our code. For instance, we used it to analyze parts of the training pipeline and dataset preprocessing. Although these efforts were limited, they provided insights into areas where we could improve efficiency, such as reducing redundant operations in data transformations. However, we did almost no optimizations based on profiling results, as our primary focus was on functionality.


--- question 16 fill here ---




## Working in the cloud


> In the following section we would like to know more about your experience when developing in the cloud.


### Question 17


> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:
The Cloud Run API was used to deploy and manage our containerized API in a serverless environment. The Artifact Registry API allowed us to store and manage Docker images securely, so our containerized applications were always ready for deployment. We used Cloud Storage buckets to store raw datasets, processed data, and other files. These buckets were integrated with DVC (Data Version Control) to keep track of data versions and ensure consistency across experiments, making it easier to collaborate and reproduce results. 
Vertex AI API provided tools for working with machine learning models, such as training and testing. Although we used it only for specific tasks, it was helpful for experimenting with different approaches. 
The Cloud Build API was used to automate building Docker images and deploying them to Artifact Registry and Cloud Run. Compute Engine API was used occasionally to set up virtual machines when additional computing power was needed.


--- question 17 fill here ---












### Question 18


> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:
We used the Google Cloud Platform's Compute Engine to train and run our machine learning model. The virtual machine type we selected was n1-standard-1, which provides 1 vCPU and 3.75 GB of memory. This configuration was sufficient for our training needs during development.
The Compute Engine instance was configured with the Intel Haswell CPU platform, which makes sure that there is reliable and consistent performance during model training. Although no GPUs were attached to the instance, the lightweight nature of our model, built using DistilBERT, allowed us to complete training efficiently on the CPU. By using the Compute Engine, we could easily scale up resources if needed, though the n1-standard-1 instance proved sufficient for our use case.


--- question 18 fill here ---


### Question 19


> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

[this figure](figures/Buckets.png)

--- question 19 fill here ---


### Question 20


> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:


[this figure](figures/Repo.png)


--- question 20 fill here ---


### Question 21


> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:


[this figure](figures/CloudHistory.png)


--- question 21 fill here ---


### Question 22


> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:
We managed to train our model in the cloud using Google Cloud Platform's Compute Engine. We chose Compute Engine because it provided a flexible and cost-effective way to run our training pipeline without requiring significant configuration overhead.
We used an n1-standard-1 virtual machine with 1 vCPU and 3.75 GB of memory, running a containerized training environment. The custom Docker container included all the necessary dependencies, such as PyTorch, Hugging Face Transformers, and Optuna, to train and fine-tune our DistilBERT-based sentiment regression model. The training data was pulled from DVC storage, ensuring reproducibility and ease of data management.
By leveraging Compute Engine, we were able to run the training pipeline directly on the VM, handling both preprocessing and hyperparameter optimization within a single environment. The lightweight nature of our model allowed us to complete training efficiently without requiring GPUs. This setup provided the simplicity and control we needed to complete our training in a scalable and reliable cloud environment.




--- question 22 fill here ---










## Deployment


### Question 23


> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:
Yes, we managed to write an API for our model using FastAPI. The API serves as an interface for interacting with our machine learning model and data processing logic. We designed it to provide several endpoints, including one for analyzing sentiment (/analyze_sentiment), retrieving Reddit posts (/get_posts), and calculating average sentiment trends (/get_average_sentiment).
To implement the API, we used FastAPI’s lightweight and efficient framework, which allowed us to define endpoints easily and include features like request validation. For example, the /get_posts endpoint retrieves Reddit posts using the praw library, while the /get_average_sentiment endpoint processes these posts to calculate daily average sentiment scores by combining the model's predictions with the post metadata.
One special aspect of the API is the integration of middleware to enable Cross-Origin Resource Sharing (CORS). This was critical for seamless communication between the frontend (React app) and the backend. We also included functionality to serve the React frontend directly through FastAPI, simplifying deployment by combining both the frontend and backend into a single service.
--- question 23 fill here ---


### Question 24


> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:
We deployed our API using google cloud and the google run service. The API is built with FastAPI and was containerized using docker. We create a custom docker image that combines the API and the frontend which was made in react so that both parts work together seamlessly. The deployment was automated with GitHub Actions. The pipeline is pulling the data, building the docker image from the api.dockerfile and pushing it to Google Artifact Registry. After that the image was deployed to Cloud run with 4GB of memory and public access enabled to make it easy for everybody to use. The frontend and API also works locally and can be used there.




--- question 24 fill here ---


### Question 25


> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:
We implemented unit testing for our API to ensure that its endpoints function correctly. Using pytest, we tested key functionalities such as retrieving the root endpoint, analyzing sentiment for a given text, and fetching posts from a subreddit. For example, the /analyze_sentiment endpoint was tested to ensure it returns a positive sentiment for the input "I am happy," and the /get_posts endpoint was verified to return a list of posts with the expected attributes like title, url, created_date, and sentiment.
However, we have not yet performed any load testing to evaluate how the API handles high traffic or concurrent requests. To perform load testing, we could use Locust. With Locust, we could simulate multiple users making requests to the API simultaneously and measure metrics such as response time, throughput, and error rates. This would help us identify any bottlenecks or performance issues under heavy load.
In future iterations, load testing will be a priority to ensure the API can handle real-world usage scenarios reliably and efficiently.
--- question 25 fill here ---


### Question 26


> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:
We did not implement any monitoring tools, other than that we have access to google cloud monitoring through automatic deployment to google cloud
--- question 26 fill here ---


## Overall discussion of project


> In the following section we would like you to think about the general structure of your project.


### Question 27


> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:
> *Group member s224751 used almost all 50 credits during the project. Most of this, $37.73, came from cloud storage in the last two days. The high cost was likely because every time we pushed to GitHub, our workflow started a 'Get Data' step, which pulled data from the bucket and increased the read rate.*
>*The next highest cost was the Artifact Registry, which used about $2. This happened because we often pushed large Docker images to it.*
>*Overall, working in the cloud was helpful, but it also got expensive quickly. If we had set up things better, like reducing how often data was pulled or how many large files were uploaded, we could have saved some credits.*
>*I do not believe other group members used anything. This is because the billing account was linked to s224751.*


--- question 27 fill here ---


### Question 28


> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:


We made a frontend for the API so users can input a ticker symbol for a stock. This returns a graph of the stock price and sentiment on a daily average coming from Reddit for the last month. It was implemented using the node.js extension react since it's what we are most familiar with in terms of frontend languages. 


We also implemented drift detection and implemented it into the api, so the drift report is available at /report. The report compares reference data that is pulled at the time of running from Reddit, with evaluation data for the model. Both datasets are from the r/RobinHoodPennyStocks subreddit


--- question 28 fill here ---


### Question 29


> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

This workflow diagram illustrates our machine learning pipeline, integrating both local and cloud components for efficient development and deployment.
The pipeline begins with data sourced from Kaggle, which is stored in GCP Buckets for centralized management. Data is pulled locally using DVC for preprocessing and training, then pushed back to the cloud for deployment. Furthermore, we use DVC for storing our data, and this is also used in our workflows to obtain the same data all around. Between the cloud and local components, we use Git to maintain version control, while also having workflows that ensure everything is up-to-date. Hydra and typing are also used in our code for maintaining structure and good coding practice.
The model is deployed using Cloud Run, which retrieves its Docker image from the Artifact Registry. This deployment ensures scalability and seamless execution in the cloud. We also run our training in Compute Engine, which uses WandB for model registry.
We then also use Locust to stress test our application, making sure we can handle a few users. In addition, also analyzing the response time for our applications. We also monitor our buckets using monitoring within cloud storage and making sure data is not changed.
By combining these tools and processes, this pipeline ensures a robust workflow for developing, deploying, and scaling machine learning models while maintaining reproducibility and efficiency.


### Question 30


> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:
One of the biggest challenges in our project was getting all components to work together seamlessly in the cloud. Since our application involved multiple integrated parts—a backend API (FastAPI), a machine learning model (DistilBERT), a frontend (React), and cloud services (GCP’s Cloud Run and Artifact Registry). 
Integrating data storage and retrieval also was an issue. 
Pulling preprocessed datasets via DVC in a cloud environment sometimes conflicted with local workflows. We tackled this by thoroughly testing the DVC pull commands in the GitHub Actions pipeline and ensuring all necessary data files were available before deployment.
While unit testing with pytest was a crucial part of our development process, we faced significant challenges when integrating it with GitHub Actions. One of the recurring issues was that we initially didn't follow a proper branching strategy, which caused conflicts and frequent pipeline failures during development.
Since all our changes were pushed directly to the main branch, every commit triggered the CI/CD pipeline, including the pytest tests. When multiple team members were pushing changes simultaneously or making rapid updates, incomplete or buggy code often broke the tests. This not only slowed down our progress but also made debugging more difficult because the pipeline failures were tied to multiple overlapping changes.








--- question 30 fill here ---


### Question 31


> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:
Student s224743 was in charge of developing, tests and coverage for the codebase, setting up  hydra to manage hyperparameters, made a github workflow for the application and set up our cloud trigger for the artifacts.
Student s224803 was in charge of developing, Set up logging, implemented unit testing in different files, made github action workflows for different parts of the application, set up docker files, set up profiling to optimize application  
Student s214625 was in charge of setting up the models and the training environment to work on HPC and google cloud, as well as WandB with optuna to log our experiments.
Student s224751 was in charge of developing, Set up workflows for Docker and the API so they can run in the cloud. I also helped set up Compute Engine instances to enable running training in the cloud. Additionally, I set up a Cloud Storage bucket. I’ve made stress tests for the frontend as well. Furthermore, I set up the project in GCP and managed the billing to keep it under control.
I (student s214636) was in charge of developing the API and accompanying tests for the API. I also made the front end for the API react/node.js.  I made docker files for the API/frontend to deploy quickly along with Google Cloud and GitHub secrets for external APIs. API integration with Reddit API to fetch current data and stock API to get live stock prices. I also made the data drifting setup and integration with the API to generate and view the report in the frontend. I also set up dvc so we could pull/push data we used to the Cloud Storage bucket(along with s224751). I did significant work on getting workflows to work both testing and building the docker images.

We have used ChatGPT to help debug our code and to better understand how to use the different tools. This was done because we realized that some of the documentation on the course page was very outdated in some cases. Additionally, we used GitHub Copilot to help write some of our code.


--- question 31 fill here ---





