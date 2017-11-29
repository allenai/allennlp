The demo included with AllenNLP is built using NPM so we can compile our assets
and code before it ends up on the client's machine.  This way, less data needs
to flow over the wire, less computation needs to take place on the client's
machine, and best practices can be enforced at compile time by automated
tools.

## Building and running the demo

First, make sure you have a relatively new version of `npm` installed on your
system.  If you are on a Mac, you can install `npm` with `brew install node`.

```
# npm -v
5.3.0
```

Next, you will need to install the dependencies specified in `package.json`.
You only need to run this once, or whenever dependencies are updated.  This
will install your dependencies into the newly created `node_modules` subfolder.

```
npm install
```

Now you can build the application.

```
npm run build
```

Built assets are placed in the `build` subfolder, which the Sanic server is
configured to use.

Now to run the demo, you can `cd` to the root AllenNLP directory and run the
following.

```
python -m allennlp.run serve
```

You may need to force refresh your web browser.

If you are developing the Javascript, then you will want to run that portion
through npm itself.  This way the browser will refresh after each code change.

```
npm start
```
