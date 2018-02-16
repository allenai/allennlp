FROM allennlp/allennlp:v0.3.0

WORKDIR /stage

RUN pip install google-cloud-storage
COPY experiments/ experiments/
COPY test.py test.py

#WORKDIR /experiments/
#RUN ls -la


# Add model caching
#ARG CACHE_MODELS=false
#RUN ./scripts/cache_models.py

# Optional argument to set an environment variable with the Git SHA
#ARG SOURCE_COMMIT
#ENV SOURCE_COMMIT $SOURCE_COMMIT

#LABEL maintainer="allennlp-contact@allenai.org"

EXPOSE 8000
CMD ["/bin/bash"]
