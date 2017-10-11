# Releasing

## Cutting a release

AllenNLP uses `major.minor.revision` versioning.  Most releases will uptick the `revision` number by 1.

To create a new release, check out the SHA for which you want to cut a release and push an annotated tag to GitHub.

```
git tag -a v0.2.0 -m "v0.2.0"
git push origin v0.2.0
```

## Updating the web demo

The web demo is deployed by sending a docker image to our kubernetes cluster.  When creating a new release, you should try the application locally before deploying it.

```
docker pull allennlp/allennlp:latest
docker run -p 8000:8000 -it allennlp/allennlp:latest allennlp/run serve
```

After testing, you can cut a new release of the web demo.  Versioning for the web demo is separate for the library, because typically we will have many more releases for our web demo than our library.  Web demo releases are annotated by date in `yyyy.mm.dd-i` format, where i is a release number that starts at 0 and goes up for each release on a given day.

```
docker tag allennlp/allennlp allennlp/webdemo:2017-09-05-0
docker push allennlp/webdemo:2017-09-05-0
```

If you get an "denied:" error at the last step, make sure to `docker login`.

Now you can edit `kubernetes-webdemo.yaml` with the latest version, and update the web demo with the following command.

```
kubectl apply -f kubernetes-webdemo.yaml
```

You should commit your modifications to `kubernetes-webdemo.yaml` and push them to master after you deploy.

## Releasing a new version on pip
