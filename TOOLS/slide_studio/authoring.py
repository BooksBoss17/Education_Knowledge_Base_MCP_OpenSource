"""Animation authoring guidance shared by the model worker and the MCP edit tool."""

ANIMATION_GUIDANCE = (
    "Follow the teacher's instructions for the slide. By default, use editable presets "
    "for text reveals and simple entrances. For a continuous demonstration, you can "
    "author self-contained HTML, SVG, Canvas or WebGL with JavaScript within the bounds "
    "of an object with type='component'. Put its markup in html and declare its action names in actions. "
    "Use a timeline step with effect='component', trigger='click', target and action. "
    "Handle the studio-action event in the component (event.detail is the action name). "
    "One Next step can animate multiple objects together and stop where the class should "
    "observe or predict. Use the slide timeline for playback of passive demonstrations. "
    "Add on-canvas controls when learners need to change conditions or manipulate the experiment."
)
