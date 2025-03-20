# Cursor Settings - Rules for AI

Copy and paste these rules into your Cursor settings under 'Rules for AI':

```
Always greet me with my name - Ricsi.

Review .cursorrules files and /docs, every message should reference the cursorrules.

It is very important to have a working memory.
!!Always check these files for current project state before any work!!:

## Required Pre-Work Checklist
- [ ] Review /docs/plan.md - Main project plan and task tracking
- [ ] Review /docs/plan-podcast.md - Podcast feature specific planning
- [ ] Output plan updates before starting work
- [ ] Reference plan number in all communications

## Project Structure Rules
- All models should be in models/ and predictors in predictors/
- Follow consistent naming conventions across all files
- Use semantic versioning for all changes

## Development Guidelines
- Review docs/composer-history for current and previous tasks
- Every run should use composer history and .plan
- Reference the .cursorrules file in communications
- Be surgical in code changes, only modify what's necessary
- Get explicit permission before deleting files
- Review large deletions before committing

## Documentation Standards
- Keep documentation structure consistent
- Use relative links from repository root
- Maintain changelog entries
- Always run a command to get the current date and time
```

## Implementation Notes
- Replace any remaining placeholders with actual values
- Verify all file paths are correct
- Test all documentation links
- Follow semantic versioning
