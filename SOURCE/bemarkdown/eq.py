from __future__ import annotations

from dataclasses import dataclass, field


class EqParseError(ValueError):
    pass


@dataclass
class EqText:
    value: str


@dataclass
class EqCommand:
    name: str
    modifiers: list[str] = field(default_factory=list)
    args: list[EqSequence] = field(default_factory=list)
    source: str = ""


@dataclass
class EqSequence:
    items: list[EqText | EqCommand]


@dataclass
class EqConversion:
    latex: str
    warnings: list[str]
    approximate: bool
    ast: EqSequence


class EqParser:
    """Tokenizer and recursive parser for Word EQ fields.

    Commas and semicolons split arguments only at the current parenthesis level.
    Switches are parsed structurally rather than with global replacements.
    """

    def __init__(self, source: str):
        stripped = source.strip()
        if stripped[:2].upper() == "EQ":
            stripped = stripped[2:].lstrip()
        self.source = stripped
        self.pos = 0

    def parse(self) -> EqSequence:
        sequence, delimiter = self._sequence(set())
        if delimiter is not None or self.pos != len(self.source):
            raise EqParseError(f"Unexpected delimiter at offset {self.pos}")
        return sequence

    def _sequence(self, stops: set[str]) -> tuple[EqSequence, str | None]:
        items: list[EqText | EqCommand] = []
        text: list[str] = []
        literal_parenthesis_depth = 0
        while self.pos < len(self.source):
            char = self.source[self.pos]
            if char in stops and literal_parenthesis_depth == 0:
                if text:
                    items.append(EqText("".join(text)))
                self.pos += 1
                return EqSequence(items), char
            if char == "\\":
                if text:
                    items.append(EqText("".join(text)))
                    text = []
                items.append(self._command())
            else:
                # Ordinary parentheses belong to displayed math. Their closing
                # delimiters and inner commas must not end the EQ switch's
                # argument; nested switch arguments are consumed by _command.
                if char == '(':
                    literal_parenthesis_depth += 1
                elif char == ')' and literal_parenthesis_depth:
                    literal_parenthesis_depth -= 1
                text.append(char)
                self.pos += 1
        if text:
            items.append(EqText("".join(text)))
        return EqSequence(items), None

    def _command(self) -> EqCommand:
        start = self.pos
        self.pos += 1
        name_start = self.pos
        while self.pos < len(self.source) and self.source[self.pos].isalpha():
            self.pos += 1
        name = self.source[name_start : self.pos].lower()
        if not name:
            self.pos += self.pos < len(self.source)
            return EqCommand("literal", source=self.source[start : self.pos])

        modifiers: list[str] = []
        while True:
            saved = self.pos
            while self.pos < len(self.source) and self.source[self.pos].isspace():
                self.pos += 1
            if self.pos >= len(self.source) or self.source[self.pos] != "\\":
                break
            self.pos += 1
            mod_start = self.pos
            while self.pos < len(self.source) and self.source[self.pos].isalpha():
                self.pos += 1
            modifier = self.source[mod_start : self.pos].lower()
            if not modifier:
                self.pos = saved
                break
            while self.pos < len(self.source) and self.source[self.pos].isspace():
                self.pos += 1
            number_start = self.pos
            if self.pos < len(self.source) and self.source[self.pos] in "+-":
                self.pos += 1
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
            if self.pos > number_start:
                modifier += self.source[number_start : self.pos]
            modifiers.append(modifier)

        while self.pos < len(self.source) and self.source[self.pos].isspace():
            self.pos += 1
        args: list[EqSequence] = []
        if self.pos < len(self.source) and self.source[self.pos] == "(":
            self.pos += 1
            while True:
                arg, delimiter = self._sequence({",", ";", ")"})
                args.append(arg)
                if delimiter == ")":
                    break
                if delimiter is None:
                    raise EqParseError(f"Unclosed EQ command {name!r}")
        return EqCommand(
            name=name,
            modifiers=modifiers,
            args=args,
            source=self.source[start : self.pos],
        )


class EqLatexRenderer:
    def __init__(self):
        self.warnings: list[str] = []
        self.approximate = False

    def render(self, sequence: EqSequence) -> EqConversion:
        latex = self._sequence(sequence).strip()
        if not latex:
            raise EqParseError("EQ field produced empty output")
        return EqConversion(latex, self.warnings, self.approximate, sequence)

    def _sequence(self, sequence: EqSequence) -> str:
        return "".join(self._item(item) for item in sequence.items)

    def _item(self, item: EqText | EqCommand) -> str:
        if isinstance(item, EqText):
            return item.value.strip() if item.value.isspace() else item.value
        args = [self._sequence(arg).strip() for arg in item.args]
        name = item.name
        mods = item.modifiers
        if name == "literal":
            return item.source
        if name == "f" and len(args) >= 2:
            return rf"\frac{{{args[0]}}}{{{args[1]}}}"
        if name == "r":
            if len(args) >= 2:
                return rf"\sqrt[{args[0]}]{{{args[1]}}}"
            if args:
                return rf"\sqrt{{{args[0]}}}"
        if name == "i":
            operator = "\\int"
            if "su" in mods:
                operator = "\\sum"
            elif "pr" in mods:
                operator = "\\prod"
            lower = args[0] if len(args) > 0 else ""
            upper = args[1] if len(args) > 1 else ""
            body = args[2] if len(args) > 2 else ""
            limits = (rf"_{{{lower}}}" if lower else "") + (
                rf"^{{{upper}}}" if upper else ""
            )
            return f"{operator}{limits} {body}".strip()
        if name in {"su", "pr"}:
            operator = "\\sum" if name == "su" else "\\prod"
            lower = args[0] if len(args) > 0 else ""
            upper = args[1] if len(args) > 1 else ""
            body = args[2] if len(args) > 2 else ""
            return f"{operator}_{{{lower}}}^{{{upper}}} {body}".strip()
        if name == "s":
            body = args[-1] if args else ""
            up = any(mod.startswith("up") for mod in mods)
            down = any(mod.startswith("do") for mod in mods)
            if up and not down:
                return rf"^{{{body}}}"
            if down and not up:
                return rf"_{{{body}}}"
            self._warn(item, "EQ \\s movement approximated as a script")
            return rf"^{{{body}}}"
        if name == "b":
            body = args[0] if args else ""
            left, right = "(", ")"
            joined = " ".join(mods)
            if "bc" in joined or "lc{" in joined:
                left, right = "\\{", "\\}"
            elif "br" in joined or "lc[" in joined:
                left, right = "[", "]"
            return rf"\left{left}{body}\right{right}"
        if name == "a":
            rows = " \\\\ ".join(args)
            return rf"\begin{{array}}{{c}}{rows}\end{{array}}"
        if name == "o":
            if len(args) >= 2:
                self._warn(item, "EQ overstrike approximated with \\overset")
                return rf"\overset{{{args[1]}}}{{{args[0]}}}"
            return args[0] if args else ""
        if name == "x":
            body = args[0] if args else ""
            joined = " ".join(mods)
            self._warn(item, "EQ border switch approximated in LaTeX")
            if "to" in joined:
                return rf"\overline{{{body}}}"
            if "bo" in joined:
                return rf"\underline{{{body}}}"
            return rf"\boxed{{{body}}}"
        if name == "d":
            self._warn(item, "EQ displacement/spacing preserved approximately")
            return "".join(args)
        if name == "l":
            return "".join(args)
        self._warn(item, f"Unsupported EQ switch \\{name} preserved approximately")
        content = ",".join(args)
        return rf"\operatorname{{EQ_{name}}}\left({content}\right)"

    def _warn(self, item: EqCommand, message: str) -> None:
        self.approximate = True
        self.warnings.append(f"{message}: {item.source}")


def convert_eq(source: str) -> EqConversion:
    return EqLatexRenderer().render(EqParser(source).parse())
