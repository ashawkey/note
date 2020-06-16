# LaTeX

### Compilers

* Distribution
  * `MiKTeX`: Windows
  * `Tex Live`: Linux

* Command

  ```bash
  latex test.tex # test.dvi
  pdflatex test.tex # test.pdf
  latexmk -pdf test.tex # test.pdf, with cross-references
  ```

  

### Basics

* `\documentclass[fontsize, papersize, ...]{document type}`

  * parameters:

    `twoside` : print in two side

    `twocolumn`: two column format

  * document type:

    `article`: default

    `letter`: for letters.

    `beamer`: beamer.

* `\\`

  break the lines

* `%`

  comments

* `~`

  Unbreakable space

```latex
\documentclass[12pt, letterpaper]{article} % [10pt, A4, twoside] by default

\title{test}
\date{2020-03-28}
\author{haw}

% do not count equation numbers
\usepackage{amsmath}

\begin{document}
	\maketitle
	\newpage
	Hello World!
	% Sections
	\section{Sections}
	\subsection{Subsection}
	\subsubsection{Subsubsection}
	% Maths
	\section{Maths}
	\begin{equation}
		1 + 2 = 3
	\end{equation}
	% Figures
	\section{Figures}
	\begin{figure}[h] % locations: h, t, b, p[!]
		\includegraphics[width=\linewidth]{picture.jpg}
		\caption{Nice boat.}
		\label{fig:boat}
	\end{figure}
	% Tables
    \section{Tables}
    \begin{table}
    	\caption{The table.}
    \end{table}
    % Lists
    \section{Lists}
    \begin{itemize} % unordered
    	\item one
    	\item two
    \end{itemize}
\end{document}
```

### Mathematics



### Bibtex

```latex
\bibliography{ref} % ref.bib
\bibliographystyle{ieeetr} % ieeetr.bst
```

##### `.bib` file

```bibtex
@book{DUMMY:1,
AUTHOR="Who",
TITLE={"Chat"},
YEAR="3200",
}
```

##### `.bst` file

Bibliography style file.



### Packages

Most packages are installed by default.

You can find them on [CTAN](https://ctan.org/).

* `tikz`: graphic elements

  draw images in pdf.

* `hyperref`: hyper links

  ```latex
  \usepackage{hyperref}
  \hypersetup{
  	colorlinks=true,
  	linkcolor=blue,
  	filecolor=magenta,
  	urlcolor=cyan,
  }
  \urlstyle{same}
  
  This is the \hyperlink{tgt}{link}.
  This is the \hypertarget{tgt}{target}.
  ```

  

### Formatting

##### Units

```
pt, mm, cm, in, ex, em, mu
```

##### Lengths

```latex
\baselineskip
\columnsep
\columnwidth
\linewidth
\paperwidth
\paperheight
\parindent
\parskip

% set
\setlength{\columnsep}{1in}
```

##### Font size

```latex
This is really {\tiny tiny}.
\tiny
\scriptsize
\footnotesize
\small
\normalsize
\large
\Large
\LARGE
\huge
\Huge
```



### Beamer

```latex
\documentclass{beamer}

\title{Sample title}
\author{Anonymous}
\institute{Overleaf}
\date{2014}

\begin{document}
% title
\frame{\titlepage}
% page 1
\begin{frame}
\frametitle{Sample frame title}
This is a text in the first frame. This is a text in the first frame. This is a text in the first frame.
\end{frame}
\end{document}
```





### Commands

```latex
% macros
\newcommand{\R}{\mathbb{R}}
use \R.
\newcommand{\bb}[1]{\mathbb{#1}}
use \bb{R}.
% insert file.tex here
\input{file}
```



### Write Class(.cls) & Package(.sty)

> The basic rule is that if your file contains commands that control the look of the logical structure of a special type of document, then it's a class. Otherwise, if your file adds features that are independent of the document type, i.e. can be used in books, reports, articles and so on; then it's a package.

##### Package

```latex
\NeedsTeXFormat{LaTeX2e} % version
\ProvidesPackage{example}[test]

% dependent
\RequirePackage{imakeidx}
\RequirePackage{xstring}
\RequirePackage{xcolor}

\definecolor{greycolour}{HTML}{525252}
\definecolor{sharelatexcolour}{HTML}{882B21}
\definecolor{mybluecolour}{HTML}{394773}
\newcommand{\wordcolour}{greycolour}

\DeclareOption{red}{\renewcommand{\wordcolour}{sharelatexcolour}}
\DeclareOption{blue}{\renewcommand{\wordcolour}{mybluecolour}}
\DeclareOption*{\PackageWarning{examplepackage}{Unknown ‘\CurrentOption’}}
\ProcessOptions\relax
```



##### Class

```latex
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{exampleclass}[2014/08/16 Example LaTeX class]

\newcommand{\headlinecolor}{\normalcolor}
\LoadClass[twocolumn]{article}
\RequirePackage{xcolor}
\definecolor{slcolor}{HTML}{882B21}


\DeclareOption{onecolumn}{\OptionNotUsed}
\DeclareOption{green}{\renewcommand{\headlinecolor}{\color{green}}}
\DeclareOption{red}{\renewcommand{\headlinecolor}{\color{slcolor}}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\renewcommand{\maketitle}{%
    \twocolumn[%
        \fontsize{50}{60}\fontfamily{phv}\fontseries{b}%
        \fontshape{sl}\selectfont\headlinecolor
        \@title
        \medskip
        ]%
}

\renewcommand{\section}{%
    \@startsection
    {section}{1}{0pt}{-1.5ex plus -1ex minus -.2ex}%
    {1ex plus .2ex}{\large\sffamily\slshape\headlinecolor}%
}

\renewcommand{\normalsize}{\fontsize{9}{10}\selectfont}
\setlength{\textwidth}{17.5cm}
\setlength{\textheight}{22cm}
\setcounter{secnumdepth}{0}
```

Then use it in a latex file:

```latex
\documentclass[red]{exampleclass}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}

\usepackage{blindtext}

\title{Example to show how classes work}
\author{Team Learn ShareLaTeX}
\date{August 2014}

\begin{document}

\maketitle

\noindent
Let's begin with a simple working example here.

\blindtext

\section{Introduction}

The Monty Hall problem...

\section{The same thing}

The Monty...
```



