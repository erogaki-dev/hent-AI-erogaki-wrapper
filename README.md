# hent-AI-erogaki-wrapper

## Building the Docker Image

### 1. Models

**Models aren't included with this repository, since we're unsure about their licensing.**

Get the following models and put them in the `models` directory:

- `hent-AI model 268`
- `25-11-2019 Fatal Pixels`

You can find links to them [here](https://github.com/erogaki-dev/hent-AI/blob/master/README.md#the-model).

Your `models` directory should then have the following subdirectories:

- `hent-AI model 268/`
- `25-11-2019 Fatal Pixels/`

### 2. `docker image build`

Then just build the docker image, when you're in the root directory of this repository, using the following command:

```
docker image build -t hent-ai-erogaki-wrapper .
```

## Running a Docker Container

Once you build the Docker image, you can just run a container like this:

```
docker run -it --network=host hent-ai-erogaki-wrapper
```
