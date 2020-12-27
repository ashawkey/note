# LaTeX tutorial

### Compiler

- **LaTeX** 

  only support `.eps, .ps` image formats.

  

- **pdfLaTeX** (default choice for English)

  supports `.png, .jpg, .pdf` image formats. 

  

- **XeLaTeX & LuaLaTeX** (if use Chinese)

  UTF-8 support. 

  supports `.png, .jpg, .pdf` image formats. 

  XeLaTeX supports `pstricks`(draw images with code), but LuaLaTeX doesn't.



### Misc.

* document class

  ```latex
  \documentclass{article}
  \documentclass[a4paper,10pt]{article} % 10pt fontsize
  \documentclass[a4paper,10pt,twoside]{article} % two-sided
  ```

* page margin

  ```latex
  \usepackage{fullpage} % use 1in margin
  
  % or
  \usepackage[top=1in,bottom=1in,left=1.25in,right=1.25in]{geometry}
  ```

* explicit new paragraph

  ```latex
  paragraph1 \par
  paragraph2
  
  % same to
  paragraph1
  
  paragraph2
  ```

* show frame

  ```latex
  \usepackage{showframe}
  \renewcommand*\ShowFrameColor{\color{red}}
  ```

  

* define new command

  ```latex
  % \newcommand{cmd}[args][opt]{def}
  \newcommand{\water}{H$_2$O}
  
  the formula of water is \water.
  \water\ is the formula of water. % \space is needed.
  
  % \renewcommand{cmd}[args][opt]{def}, cmd must have been defined.
  % usually used to change default behaviour.
  \renewcommand{\familydefault}{\sfdefault}
  ```

* space

  ```latex
  % manual adjust
  \vspace{-1in}
  ```

  

* text justification:

  ```latex
  \begin{center} ... \end{center}
  \begin{flushright} ... \end{flushright}
  \begin{flushleft} ... \end{flushleft}
  ```

* paragraph indentation

  ```latex
  \noindent % no indent for this paragraph
  ```

* text decorations

  ```latex
  \textbf{text}
  \underline{text}
  \textit{text}
  \texttt{text} % type
  ```

* font size

  ```latex
  \tiny{text}
  \scriptsize{text}
  \footnotesize{text}
  \small{text}
  \normalsize{text}
  \large{text}
  \Large{text}
  \LARGE{text}
  \huge{text}
  \Huge{text}
  ```

* font family

  ```latex
  % set default family
  \renewcommand{\familydefault}{\sfdefault}
  ```

* page size

  ```latex
  \usepackage[a4paper, total={6in, 8in}]{geometry}
  ```

* list

  ```latex
  % unordered
  \begin{itemize}
    \item one
    \item two
  \end{itemize}
  
  % ordered
  \begin{enumerate}
    \item first
    \item second
  \end{enumerate}
  ```

* marks

  ```latex
  \usepackage{amssymb}
  \checkmark
  
  % or:
  \usepackage{pifont}
  \newcommand{\cmark}{\ding{51}}
  \newcommand{\xmark}{\ding{55}}
  ```

  

### Mathematics

##### Mode

* Inline: `$inline math$`

* Display: 

  ```latex
  % unnumbered
  \[E = mc^2\]
  
  \begin{equation*} % asterisk = no number
  E = mc^2
  \end{equation*}
  
  % numbered
  \begin{equation}
  E = mc^2
  \end{equation}
  ```

##### Align (package amsmath)

```latex
% auto-wrap
\begin{multline}
p(x) = 3x^6 + 14x^5y + 590x^4y^2 + 19x^3y^3\\ 
- 12x^2y^4 - 12xy^5 + 2y^6 - a^3b^3
\end{multline}

% use & to align
\begin{align*} 
2x - 5y &=  8 \\ 
3x + 9y &=  -12
\end{align*}

% just center multiple formulas
\begin{gather*} 
2x - 5y =  8 \\ 
3x^2 + 9y =  3a + c
\end{gather*}
```



##### Matrices

```latex
% () parenthesis
\begin{pmatrix}
1 & 2 & 3\\
a & b & c
\end{pmatrix}

% [] bracket
\begin{bmatrix}
1 & 2 & 3\\
a & b & c
\end{bmatrix}

% {} braces
\begin{Bmatrix}
1 & 2 & 3\\
a & b & c
\end{Bmatrix}

% inline
$\begin{pmatrix}
  a & b\\ 
  c & d
\end{pmatrix}$ 

$\big(\begin{smallmatrix}
  a & b\\
  c & d
\end{smallmatrix}\big)$
```



##### spacings

```latex
\! % -3/18 quad

% normal spacing

\, \; \: % 3,4,5/18 quad
\  % space in normal text
\quad % space equal to current font size 
\qquad % 2 quad
```



##### fonts

```latex
\usepackage{amssymb} 
\usepackage{amsfont}

\begin{align*}
RQSZ \\
\mathcal{RQSZ} \\
\mathfrak{RQSZ} \\
\mathbb{RQSZ}
\end{align*}
```



### Bibliography



### Figures

```latex
\usepackage{graphicx}

% simply insert
\includegraphics{universe.png}

% size
\includegraphics[scale=1.5]{lion-logo}
\includegraphics[width=\textwidth]{lion-logo}
\includegraphics[width=3cm, height=4cm]{lion-logo}

% positioning
% h[ere], t[op], b[ottom], p[age]
% h!: override and force here.
\begin{figure}[h]
	\centering % default is left
	\includegraphics[width=8cm]{Plot}
\end{figure}

% wrap figure alignment {l}, {r}
\begin{wrapfigure}{l}{0.25\textwidth} % wrap image in text
    \centering 
    \includegraphics[width=0.25\textwidth]{contour}
\end{wrapfigure}

% caption & label
\begin{figure}[h]
    \centering
    \includegraphics[width=0.25\textwidth]{mesh}
    \caption{a nice plot}
    \label{fig:mesh1}
\end{figure}

% subfigures
\begin{figure}[h]

\begin{subfigure}{0.5\textwidth}
	\includegraphics[width=0.9\linewidth, height=5cm]{lion-logo} 
	\caption{Caption1}
	\label{fig:subim1}
\end{subfigure}

\begin{subfigure}{0.5\textwidth}
	\includegraphics[width=0.9\linewidth, height=5cm]{mesh}
	\caption{Caption 2}
	\label{fig:subim2}
\end{subfigure}

\caption{Caption for this figure with two images}
\label{fig:image2}
\end{figure}
```



### Tables

```latex
% regular table
\begin{table}[h]
    \small % fontsize
    \centering % centering
    \begin{tabular}{ccc} % 3 columns, no vertical lines
        \shline % thicker horizontal line
        Method & SSC mIoU(\%) & SC IoU(\%) \\
        \hline% horizontal line
        Baseline (Euclidean) & - & - \\
        Cosine similarity & - & - \\
        SWFP w/o supervision & - & - \\
        SWFP w/ supervision & - & - \\
        \shline
    \end{tabular}
    \caption{Ablation study on the type of Feature Propagation Modules.}
    \label{tab:ablation_fp}
\end{table}

% set column width explicitly
% p{width}	Top-aligned cells width fixed width
% m{width}	Middle-aligned cells width fixed width
% b{width}	Bottom-aligned cells with fixed width

\begin{tabular}{p{0.1\textwidth}p{0.8\textwidth}}
\end{tabular}

% multirow
\begin{tabular}{p{0.2\textwidth}>{\centering}p{0.2\textwidth}>{\centering}p{0.2\textwidth}>{\centering\arraybackslash}p{0.2\textwidth}}
\hline
\multirow{2}{*}{Country}&\multicolumn{2}{c}{Population}&\multirow{2}{*}{Change (\%)}\\\cline{2-3}
&2016&2017&\\
\hline
\end{tabular}

% multicol
```





### Errors

* `undefined control sequence`

  one of the commands is not defined. e.g. `\R`, `\Zlpha`

  



### Lengths

##### Units

* pt: point, 1pt = 0.3515mm
* mm: milimeter
* cm: centimeter
* in: inch
* ex: height of an 'x' in current font
* em: width of an 'M' in current font.

##### Constants

```latex
% two-column 
\columnsep
\columnwidth

% line
\linewidth
\baselineskip % vertical space between lines

% the whole paper
\paperwidth
\paperheight

% paragraph
\parindent 
\parskip 

% only the textarea
\textwidth
\textheight

% margin
\topmargin
\evensidemargin
\oddsidemargin
```

##### Example

```latex
% set figure length
\includegraphics[width=15ex]{figs/tmp.png
\includegraphics[width=\textwidth]{figs/tmp.png} % auto scale

% setlength command
\seglength{\columnsep} {1in}
```



