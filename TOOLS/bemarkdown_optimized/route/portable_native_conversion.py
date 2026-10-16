"""Native phase of the frozen route, with explicit installation paths."""
from pathlib import Path
from bootstrap import write


def convert_native(source, output, core_root, *, mcp_root, flash_patch):
    from isolated_native_backends import require_native_backend_activation, close_active
    require_native_backend_activation()
    from bemarkdown.production import convert_document
    from bemarkdown.docx_image_content import DocxImageContentAdapter
    from bemarkdown.pdf.table_staged_runtime import StagedTableEvidenceProvider
    from flash_text_resolver import pdf_factory
    mcp_root = Path(mcp_root)
    agent_root = Path(output).parent / 'native-agent'
    models = mcp_root / 'MODELS'
    configuration = mcp_root / 'TOOLS/bemarkdown/bemarkdown.toml'

    def factory(**kwargs):
        kwargs['table_evidence_provider_factory'] = lambda work: StagedTableEvidenceProvider(
            work, models_root=models, config_path=configuration,
            mcp_root=mcp_root, before_stages=close_active)
        return pdf_factory(patch=flash_patch, agent_root=agent_root,
                           native_local_first=Path(source).suffix.lower()=='.pdf', **kwargs)

    figures = DocxImageContentAdapter(models_root=models, config_path=configuration,
        mcp_root=mcp_root, pdf_runtime_factory=factory, batch_region_jobs=True)
    result = convert_document(source, output, models_root=models, mcp_root=mcp_root,
        config_path=configuration, debug=True, formula_ocr='auto',
        docx_image_route='original', replace=False, figure_text_adapter=figures,
        pdf_runtime_factory=lambda: factory(models_root=models,
            config_path=configuration, mcp_root=mcp_root))
    write(Path(output) / 'conversion-pointer.json', result.to_dict())

