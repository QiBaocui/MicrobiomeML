#!/usr/bin/env Rscript

library(optparse)

script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(script_arg)) {
  dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
} else {
  normalizePath(getwd())
}
option_list <- list(
  make_option(c("-o", "--outdir"), type = "character", default = NULL),
  make_option(c("-g", "--group"),type = "character",default=NULL),
  make_option(c("-d", "--abun"),type = "character",default=NULL)
)

opt_parser<-OptionParser(option_list=option_list)
opt<-parse_args(opt_parser)

outdir <- opt$outdir
group <- opt$group
abun <- opt$abun

if (is.null(outdir)) stop("[Error] 请指定输出目录 -o/--outdir", call. = FALSE)
if (is.null(group)) stop("[Error] 请指定分组文件 -g/--group", call. = FALSE)
if (is.null(abun)) stop("[Error] 请指定特征矩阵 -d/--abun", call. = FALSE)

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
runner <- file.path(script_dir, "run_model.R")
if (!file.exists(runner)) stop("[Error] 未找到模型入口脚本: ", runner, call. = FALSE)

rscript <- file.path(R.home("bin"), "Rscript")
if (!file.exists(rscript)) stop("[Error] 当前 R 环境中未找到 Rscript: ", rscript, call. = FALSE)

output_file <- paste0(outdir, "/predict.sh")
file_conn <- file(output_file, "w")
writeLines("#!/usr/bin/env bash", file_conn)

for (i in 1:10) {
  ip <- sample(54321:65535, 1)
  args <- c(
    shQuote(runner),
    "-o", shQuote(file.path(outdir, i)),
    "-g", shQuote(group),
    "-d", shQuote(abun),
    "-t", "0.8", "-S", "proportional", "-f", "5",
    "-K", "radial", "-T", "FALSE", "-C", "TRUE",
    "--svm_tune_folds", "3", "-m", "3000", "-n", "5",
    "-p", as.character(ip)
  )
  cmd <- paste(shQuote(rscript), paste(args, collapse = " "))
  writeLines(cmd, file_conn)
}
close(file_conn)

Sys.chmod(output_file, mode = "0755")
message("[Done] 已生成: ", output_file)
