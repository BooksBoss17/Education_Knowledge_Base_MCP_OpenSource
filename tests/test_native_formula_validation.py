from __future__ import annotations

import pytest

from bemarkdown.formula import FormulaConversion, convert_mathml
from bemarkdown.validator import FormulaStructuralValidator, FormulaVerdict

VALID_LATEX = [
    "F",
    r"\theta",
    "t=0",
    "x",
    "x_1",
    "F=ma",
    r"f_{\text{洛}}=qvB",
    r"\frac{x}{y}",
    r"\sqrt{x}",
]


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("<mi>v</mi>", r"\bar{v}"),
        ("<msub><mi>v</mi><mn>1</mn></msub>", r"\bar{v_{1}}"),
        ("<mfrac><mi>x</mi><mn>2</mn></mfrac>", r"\bar{\frac{x}{2}}"),
        ("<mover><mi>v</mi><mo>¯</mo></mover>", r"\bar{\bar{v}}"),
    ],
)
def test_mathml_macrons_render_with_balanced_latex_groups(base, expected):
    mathml = f'<math><mrow><mover accent="true">{base}<mo>¯</mo></mover><mo>+</mo><mi>x</mi></mrow></math>'
    conversion = convert_mathml(mathml, component="mathml2latex")
    assert conversion.latex == expected + "+x"
    assert conversion.intermediate == mathml
    assert FormulaStructuralValidator().validate(conversion).accepted


def test_unknown_mathml_overscript_is_rejected_without_changing_its_source():
    mathml = "<math><mrow><mover><mi>v</mi><mo>→</mo></mover></mrow></math>"
    conversion = convert_mathml(mathml, component="mathml2latex")
    assert conversion.latex == r"\mover{v}^{→}"
    assert conversion.intermediate == mathml
    validation = FormulaStructuralValidator().validate(conversion)
    assert validation.verdict == FormulaVerdict.INVALID
    assert {issue.code for issue in validation.issues} == {"UNSUPPORTED_MATHML_OVERSCRIPT"}


def test_escaped_mover_text_does_not_trigger_overscript_validation():
    mathml = "<math><mover><mi>v</mi><mo>¯</mo></mover></math>"
    conversion = FormulaConversion(r"\\mover", "SUCCESS_NORMALIZED", "test", [], intermediate=mathml)
    assert FormulaStructuralValidator().validate(conversion).accepted


@pytest.mark.parametrize("latex", VALID_LATEX)
def test_validator_preserves_complete_short_and_chinese_formulas(latex):
    result = FormulaStructuralValidator().validate(
        FormulaConversion(latex, "SUCCESS_NORMALIZED", "test", [])
    )
    assert result.verdict == FormulaVerdict.VALID
    assert result.issues == ()


@pytest.mark.parametrize(
    ("latex", "issue"),
    [
        (r"\frac{}{x}", "EMPTY_NUMERATOR"),
        (r"\frac{x}{}", "EMPTY_DENOMINATOR"),
        (r"\sqrt{}", "EMPTY_RADICAND"),
        (r"x_{}", "EMPTY_SCRIPT_OPERAND"),
        (r"x^{}", "EMPTY_SCRIPT_OPERAND"),
        (r"\frac{}{5\sqrt{}}s", "EMPTY_NUMERATOR"),
        (r"\frac{}{_{I}^{}R}", "EMPTY_SCRIPT_OPERAND"),
    ],
)
def test_validator_rejects_obviously_empty_latex_operands(latex, issue):
    result = FormulaStructuralValidator().validate(
        FormulaConversion(latex, "SUCCESS_NORMALIZED", "test", [])
    )
    assert result.verdict == FormulaVerdict.SUSPICIOUS
    assert issue in {item.code for item in result.issues}
    assert "EMPTY_REQUIRED_OPERAND" in {item.code for item in result.issues}


@pytest.mark.parametrize(
    ("body", "issue"),
    [
        ("<mfrac><mrow/><mi>x</mi></mfrac>", "EMPTY_NUMERATOR"),
        ("<mfrac><mi>x</mi><mrow/></mfrac>", "EMPTY_DENOMINATOR"),
        ("<msqrt><mrow/></msqrt>", "EMPTY_RADICAND"),
        ("<msub><mi>x</mi><mrow/></msub>", "EMPTY_SCRIPT_OPERAND"),
        ("<msup><mrow/><mi>2</mi></msup>", "EMPTY_BASE"),
    ],
)
def test_validator_checks_mathml_tree_required_operands(body, issue):
    mathml = f'<math xmlns="http://www.w3.org/1998/Math/MathML">{body}</math>'
    conversion = FormulaConversion(
        "converted", "SUCCESS_NORMALIZED", "test", [], intermediate=mathml
    )
    result = FormulaStructuralValidator().validate(conversion)
    assert result.verdict == FormulaVerdict.SUSPICIOUS
    assert issue in {item.code for item in result.issues}


def test_validator_marks_empty_or_broken_mathml_invalid():
    empty = FormulaConversion(
        None,
        "EMPTY_SEMANTIC_PAYLOAD",
        "test",
        [],
        intermediate='<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow/></math>',
        semantic_state="empty",
    )
    broken = FormulaConversion(
        "x", "SUCCESS_NORMALIZED", "test", [], intermediate="<math><mi>x</math>"
    )
    validator = FormulaStructuralValidator()
    assert validator.validate(empty).verdict == FormulaVerdict.INVALID
    assert validator.validate(broken).verdict == FormulaVerdict.INVALID


def test_validator_distinguishes_punctuation_and_orphan_text_fragment():
    validator = FormulaStructuralValidator()
    punctuation = validator.validate(
        FormulaConversion(".", "SUCCESS_NORMALIZED", "test", [])
    )
    assert punctuation.verdict == FormulaVerdict.NON_FORMULA_CONTENT

    mathml = (
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>的</mtext></math>'
    )
    orphan = validator.validate(
        FormulaConversion(
            "的", "SUCCESS_NORMALIZED", "test", [], intermediate=mathml
        ),
        source_metadata={"referenced": False},
    )
    referenced = validator.validate(
        FormulaConversion(
            "的", "SUCCESS_NORMALIZED", "test", [], intermediate=mathml
        ),
        source_metadata={"referenced": True},
    )
    assert orphan.verdict == FormulaVerdict.NON_FORMULA_CONTENT
    assert referenced.verdict == FormulaVerdict.SUSPICIOUS


def test_validator_reports_unbalanced_latex_without_repairing_it():
    latex = r"\frac{x}{y"
    result = FormulaStructuralValidator().validate(
        FormulaConversion(latex, "SUCCESS_NORMALIZED", "test", [])
    )
    assert result.verdict == FormulaVerdict.INVALID
    assert "UNMATCHED_DELIMITER" in {item.code for item in result.issues}
    assert result.evidence["latex"] == latex
