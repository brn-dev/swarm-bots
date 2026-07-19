## General
Be blunt. If you think my request is flawed, tell me what's wrong and suggest something better. If you are unsure about something, ask.

## Notes
Use `AGENT_NOTES.md` to write down important notes about the code base so future instances will have it easier. When making changes relating to more than one or a few files, read the notes first so ouy have an easier time navigating the codebase. If you update the code structure or add new features, make sure to keep the notes updated.  Only note down important stuff, not simply a summary of what you did. If only minor changes were done (that do not contain someting like a gotcha), leave the notes as is.

## Comments 

Do NOT write simple comments that simple describe the next lines/the next few lines! For example: 
```python
# doing something      <-- don't do this
do_something() 
```

Only write comments when something is non-obvious or explains a decision!

## Coding style
* write clean but simple code
* don't repeat yourself!!! Do NOT produce duplicate code, structure code nicely and move functions in separate files if necessary.
* keep it adaptable/maintainable
* keep code easily readable, use descriptive/self-explanatory variable names (even if they are longer)
* always place type annotations on functions 
* Don't be overly defensive
* Do NOT make shape or similar checks if they would result in a hard pytorch error anyway

## Tests
When writing tests, check the intent - do not tailor the tests to the implementation.

## Guidance
If you are unsure about how a specific library works or how its API looks like, search the web. 

Don't care too much about backwards compatibility. It's better to implement something properly, just tell me if something breaks old stuff.  
  
See [gymnasium_autoreset.md](gymnasium_autoreset.md) for NEXT_STEP auto reset which we use. You often get confused here.
See `gymnasium_autoreset.md` or `swarmbots.learn.env_wrappers.worker_pool_async_vector_env` about how next-step auto-reset works.

## Python environment

The project uses uv and the existing virtual environment is `.venv`.

For normal Python execution and tests, invoke the virtual environment directly:

- `.\.venv\Scripts\python.exe <script>`
- `.\.venv\Scripts\python.exe -m pytest`
- `.\.venv\Scripts\python.exe -m ruff check .`

Do not activate the environment in a separate command.
Do not use `uv run` merely to execute tests or scripts.
Use `uv sync`, `uv add`, and other uv commands only when managing dependencies.

## Infos
We are using python 3.11, am planning to upgrade to 3.13 soon.  
Do NOT ask for permission to modify files within this repo.  
