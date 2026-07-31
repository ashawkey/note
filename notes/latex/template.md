# LaTeX Templates

### Article

```latex
\documentclass{article}

\usepackage[utf8]{inputenc}

\usepackage[top=1in,bottom=1in,left=1.25in,right=1.25in]{geometry} % page margin
\usepackage{amsmath,amsfonts,amssymb} % math
\usepackage{graphicx} % figures
\usepackage{hyperref} % auto hyperlinks

\usepackage{url} % usage: \url{http://}

\usepackage{caption} % subfig
\usepackage{subcaption} %subfig

\usepackage{multicol}


\title{title}
\author{author}
\date{\today}

\begin{document}
\maketitle

\section{Introduction}
\input{introduction}

\bibliographystyle{unsrt}  % unsrt.bst
\bibliography{references} 

\end{document}
```

