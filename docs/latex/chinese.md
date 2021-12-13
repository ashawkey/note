# LaTeX for Chinese

use `XeLaTeX` compiler.

```latex
\documentclass{article}

\usepackage[utf8]{inputenc} % input encoding
\usepackage{xeCJK} % CN: chinese support
\usepackage{indentfirst} % CN: indent first paragraph

\usepackage[top=1in,bottom=1in,left=1.25in,right=1.25in]{geometry} % page margin
\usepackage{amsmath,amsfonts,amssymb} % math
\usepackage{graphicx} % figures
\usepackage{hyperref} % hyperlinks

\usepackage{caption} % subfig
\usepackage{subcaption} %subfig

\renewcommand{\refname}{参考文献}
\renewcommand{\figurename}{图}
\renewcommand{\tablename}{表}

\title{题目}
\author{作者}
\date{日期}

\begin{document}

\maketitle

\section{介绍}

\bibliographystyle{ieee_fullname}
\bibliography{ref}

\end{document}
```

