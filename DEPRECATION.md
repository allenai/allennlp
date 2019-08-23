# Deprecation Policy Proposal

As AllenNLP matures we will inevitably need to deprecate old models, interfaces and features. This document provides guidelines that attempt to minimize churn. Reducing churn helps users build next generation models with tools they've already mastered.

### Terminology note

We _deprecate_ some component to communicate that a preferred alternative exists. A _deprecated_ component does not necessarily have to be _removed_ from the library.

### Goals

1. Promote backward compatibility.
2. Describe code-level and release process for both deprecation and removal.

### Non-goals

1. Eliminate all deprecation and removal.
2. Legislate. Good judgment by AllenNLP developers is much preferred.

## Decision Flow

Say you've found some old code that needs some care. You have a fix in mind, maybe already a PR. Great, let's get it merged! But let's give some thought to backward compatibility along the way.

1. Determine whether backward compatibility is even an issue.

  * __Likely safe__

    * New code
    * Internal code _TODO: How is this determined?_
    * Implementation change only

  * __Care required__

    * File/class/function names
      * Including names passed to `Registrable.register`!
    * Function signatures
    * Config API for defining models, dataset readers, etc.
      * In effect, the `__init__` (or custom `from_params`) methods for `Registrable` classes.
    * Model internals that affect shape of saved weights

2. If so, first consider how impactful the change is. Impact should be weighed against cost. A low impact change shouldn't impose much (or even any) cost on our users.

  * __Low impact__

    * Minor name changes
    * "Cleaner" APIs with no functional benefit

  * __Medium impact__

    * Name/API that is actively confusing multiple users.
      * This means Github issues, messages on user channels, etc.
    * Useful new feature

  * __High impact__

    * Major bugs

3. If the impact merits the cost, let's try to make the change in a backward compatible manner.

  * Options include:
    * When deprecating, say, a class, leave a shim to its replacement.
      * E.g., an existing class implementation might be replaced by subclassing its more general replacement.
    * A function signature can likely be changed with a keyword argument.
    * You can decorate with `Registrable.register` multiple times.
    * Features can be hidden behind flags.
      * E.g., simple boolean arguments to constructors.
    * Files can simply be forked. (for extreme cases)
      * This could occur when extending an existing model, for instance.
      * Copying a file, modifying it, and then de-duplicating can be simpler than parameterizing existing code to handle a new use-case in a purely backwards compatible manner.

  * Likely no deprecation needs to occur in this case. However, in the spirit of providing a single preferred solution, one may still mark the old component as deprecated while _not_ removing it. Usages in AllenNLP should be removed so that our warnings are not spammy. Though silencing warnings is an option, we should "eat our own dog food" or we can hardly expect our users to migrate.

4. If no backward compatible change can be made, it's adivsable to first consult with other developers.

  * Ideally they will propose an alternative solution or help mitigate the churn during the change.
  * Next proceed to __Mechanics__.

## Mechanics

1. Mark the old component as deprecated by adding a `DeprecationWarning`.
  * This should include whether the component will be removed.
  * If it must be removed, describe why in an accompanying comment. Link relevant issues.
  * [Example](https://github.com/allenai/allennlp/blob/cb9651a4c77c10cbd2d76f79b85c6453386dc229/allennlp/modules/text_field_embedders/basic_text_field_embedder.py#L141)
  * Provide the version and date when we will remove the feature if applicable.
    * We should support whichever is longest.
    * The code should live for at least one full minor version and 3 months before removal.
      * e.g., if you're committing the deprecation to master while version 0.8.5 is out, then it should live throughout version 0.9 and can first be removed in version 0.10.0.
      * In particular, this should be at least a minor release, i.e. m.n.0.
      * If this isn't possible, consult with other developers. You should have a compelling rationale.

2. Remove any AllenNLP usages of the deprecated feature to avoid warnings.
  * Suppressing warnings should be done rarely. [See here for instance.](https://github.com/allenai/allennlp/blob/9719b5c71207e642276fb1209ea1a4c8467e0792/allennlp/modules/token_embedders/embedding.py#L14)

3. Create a Github issue for the actual removal and wait for the requisite removal release.
  * Link the deprecation warning.
  * Copy over the removal release date and version that you are targeting for easy issue triage.

4. You or another developer should coordinate the removal PR such that it will go into the desired release.
  * Add a "Breaking Changes" section to the release notes.

5. Release
