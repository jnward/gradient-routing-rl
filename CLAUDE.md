NEVER IMPLEMENT FALLBACKS. If something unexpected happens in code, instead of writing logic to silently handle it, you should always default to throwing an error.

Be liberal with asserts. If any part of the code is not working as expected, this will wreck the experiment.

Use uv for package management and running python code.

Don't use code like `var = data.get(thing, 1.0)` or `var = data.get(thing, None) ... if var is None: # Fallback: use default behavior`. Instead, just throw an error instead of guessing what a variable should be: `var = data[thing]`

We always want to throw errors if any part of the state is not well defined.