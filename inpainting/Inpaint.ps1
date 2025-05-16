<#

Sub-pipeline for inpainting all images in a folder.

#>


param (
    # ImageDirectory
    [Parameter(Mandatory=$true)]
    [string]
    $input_dir,

    # Pipeline File (variant)
    [Parameter(Mandatory=$true)]
    [string]
    $process,

    # ImageDirectory
    [Parameter(Mandatory=$true)]
    [string]
    $output_dir
)

function Write-ColorOutput($ForegroundColor)
{
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output "${BOLD}${args}${RESET}"
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-ColorOutput green "Inpainting ${input_dir} with pipeline variant ${process}"

# & python inpaint_any
& python .\inpainting\Inpaint_Anything\${process} `
    --input_img ${input_dir} `
    --coords_type key_in `
    --point_coords 343 382 `
    --point_labels 1 `
    --dilate_kernel_size 5 `
    --output_dir ${output_dir} `
    --sam_model_type vit_h `
    --sam_ckpt .\inpainting\Inpaint_Anything\pretrained_models\sam_vit_h_4b8939.pth `
    --lama_config .\inpainting\Inpaint_Anything\lama\configs\prediction\default.yaml `
    --lama_ckpt   .\inpainting\Inpaint_Anything\pretrained_models\big-lama
