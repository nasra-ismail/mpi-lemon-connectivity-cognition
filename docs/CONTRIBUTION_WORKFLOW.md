# Pull-request and review workflow

All planned work should move through an issue, a branch, and a pull request before it
reaches `main`.

## Standard workflow

1. Select or create one issue with a definition of done.
2. Create a branch such as `issue-39-contribution-templates`.
3. Make only the scoped change and run the relevant validation.
4. Open a **draft** pull request early.
5. Complete the author checklist and inspect the complete **Files changed** tab.
6. Mark the pull request ready for review.
7. Resolve review conversations and wait for required checks.
8. Squash-merge only when the merge-gate checklist is complete.

The pull-request template supports both scientific review and ordinary code review.
`CODEOWNERS` identifies `@xamdoo` as the default owner. GitHub requests code-owner
review only after the file exists on the pull request's base branch and the pull
request is marked ready rather than draft.

## Protecting `main`

Templates do not prevent a direct push or an early merge. Configure a branch ruleset
or classic branch-protection rule for `main` in repository settings.

Recommended rules for this repository:

- require a pull request before merging;
- require the `test` status check to pass;
- require review conversations to be resolved;
- block force pushes and branch deletion;
- dismiss stale approvals when new commits materially change a reviewed pull request;
- require approval from a code owner once a second trusted collaborator has write
  access.

### If you are working alone

Do **not** require one approval yet: GitHub does not allow the author to approve their
own pull request, so a one-approval rule would block every solo change. Require the
pull-request workflow, passing checks, and resolved conversations, then complete both
the author and reviewer checklists yourself before merging.

### After adding a collaborator

Set required approvals to one and enable code-owner review. The reviewer should be a
person other than the pull-request author.

Official references:

- [About protected branches](https://docs.github.com/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [About code owners](https://docs.github.com/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
