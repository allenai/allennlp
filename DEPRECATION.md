# Deprecation Policy Proposal

As AllenNLP matures we will inevitably need to deprecate old models, interfaces and features. This document provides guidelines that attempt to minimize churn. Reducing churn helps users build next generation models with tools they've already mastered.

### Terminology note

We _deprecate_ some component to communicate that a preferred alternative exists. A _deprecated_ component does not necessarily have to be _removed_ from the library.

### Goals

1. Promote backward compatibility where not onerous.
2. Describe code-level and release process for both deprecation and removal.

### Non-goals

1. Eliminate all deprecation and removal.
2. Legislate. Good judgment by AllenNLP developers is much preferred.

## Decision Flow

Say you've found some old code that needs some care. You have a fix in mind, maybe already a PR. Great, let's get it merged! But let's give some thought to backward compatibility along the way.

1. Determine whether backward compatibility is even an issue.
  #### Likely safe
  * New code
  * Internal code _TODO: How is this determined?_
  * Implementation change only

  #### Care required
  * File/class/function names
    * Including names passed to `Registrable.register`!
  * Function signatures
  * Jsonnet config
  * Model internals that affect shape of saved weights

2. If so, first consider how critical the change is.
  #### Non-critical
  * Minor name changes
  * "Cleaner" APIs with no functional benefit

  #### Medium
  * Name/API that is actively confusing multiple users.
    * This means Github issues, messages on user channels, etc.
  * Useful new feature

  #### High
  * Major bugs

3. Non-critical changes we should happily avoid. Otherwise, let's try to make the change in a backward compatible manner. Options include:
  * When deprecating, say, a class, leave a shim to its replacement.
  * A function signature can likely be changed with a keyword argument.
  * You can decorate with `Registrable.register` multiple times.
  * Features can be hidden behind flags.
  * Files can simply be forked. (for extreme cases)

Likely no deprecation needs to occur in this case. However, in the spirit of TOOWTDI, one may still mark the old component as deprecated while _not_ removing it. Usages in AllenNLP should be removed so that our warnings are not spammy. Though silencing warnings is an option, we should "eat our own dog food" or we can hardly expect our users to migrate.

4. No backward compatible change can be made.

In this situation it's adivsable to consult with other developers. Hopefully they can propose an alternative solution or help mitigate the churn during the change.

## Mechanics

1. Mark the old component as deprecated by adding a `DeprecationWarning`.
  * This should include whether the component will be removed.
  * If it must be removed, describe why in an accompanying comment. Link relevant issues.
  * [Example](https://github.com/allenai/allennlp/blob/cb9651a4c77c10cbd2d76f79b85c6453386dc229/allennlp/modules/text_field_embedders/basic_text_field_embedder.py#L141)
  * Provide the version and date when we will remove the feature if applicable.
    * We should support whichever is longest.
    * The code should live for at least one full minor version and 3 months before removal.
      * e.g., if you're committing the deprecation to master while version 0.8.4 is out, then it should live throughout version 0.9 and can first be removed in version 0.10.0.
      * If this isn't possible, consult with other developers. You should have a quite compelling rationale.

2. Remove any AllenNLP usages of the deprecated feature to avoid warnings.
  * Suppressing warnings should be done rarely. See https://github.com/allenai/allennlp/blob/9719b5c71207e642276fb1209ea1a4c8467e0792/allennlp/modules/token_embedders/embedding.py#L14.

3. Wait

4. Coordinate your removal PR such that it will go into at least a minor release, i.e. m.n.0.
  * Add a "Breaking Changes" section to the release notes.

5. Release
