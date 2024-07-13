# featurization-deployment

What is Featurization?
- Featurization is computing features using a pre-trained deep learning model transfers knowledge about good features from the original domain.

Deployment issues?
- Several issues when we use the serverless architecture.
    - Install tensorflow (Will need AWS EFS)
    - API GATEWAY timeout 29 seconds
- High cost to address these issues.

How to solver?
- Use the containerization using docker and deploy in AWS ECS

### Requirements
- Python
- Docker

### NOTES

Python Packages
- To run heavy modules like tensorflow, you need to have the Debian versions like **bookwork** and **bullseye**.

### How to run
- Create a folder called **Models** inside **flask** directory.
- Download the pre-trained model from the below link.
    - https://storage.googleapis.com/tensorflow/keras-applications/convnext/convnext_xlarge_notop.h5
- Copy the downloaded model to inside the Models folder.
- Change the volume path according to your volume path in docker-compose.yaml.
- Go to featurization-deployment directory through terminal.
- Run the command: docker-compose up
- Check http://localhost:5000/inference for the server.

### How to test the server
- Go to Test folder directory thorugh terminal.
- Run the command: python test.py
- CHeck whether you are getting the success response.
