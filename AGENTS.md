## General
Be blunt. If you think my request is flawed, tell me what's wrong and suggest something better. If you are unsure about something, ask.

## Mandatory startup rule
Before doing any analysis, planning, or code changes, always read `AGENT_NOTES.md` first.
This is required for every task, without exception.

## Notes
Use `AGENT_NOTES.md` to write down notes about the code base so future instances will have it easier. If you update the code structure or add new features, make sure to keep the notes updated.

## Comments 

Do NOT write simple comments that simple describe the next lines/the next few lines! For example: 
```python
# doing something      <-- don't do this
do_something() 
```

Only write comments when something is non-obvious!

## Coding style
* write clean but simple code
* don't repeat yourself!!! Do NOT produce duplicate code, structure code nicely and move functions in separate files if necessary.
* keep it adaptable/maintainable
* keep code easily readable, use descriptive/self-explanatory variable names (even if they are longer)
* always place type annotations on functions 
* Don't be overly defensive

## Guidance
If you are unsure about how a specfic library works or how its API looks like, search the web. 

Don't care too much about backwards compatibility. It's better to implement something properly, just tell me if something breaks old stuff.
See [gymnasium_autoreset.md](gymnasium_autoreset.md) for NEXT_STEP auto reset which we use. You often get confused here.

See `gymnasium_autoreset.md` or `swarmbots.learn.env_wrappers.worker_pool_async_vector_env` about how next-step auto-reset works.

## Infos
We are using python >= 3.11, am planning to upgrade to 3.13 soon.  
