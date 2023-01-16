---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>VIOLA: Imitation Learning for Vision-Based Manipulation <br> with Object Proposal Priors</title>

<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  h3 {
    font-weight:300;
    font-size: 20px
  }
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
td {
  font-size: 20px
}
hr
  {
    border: 0;
    height: 2.5px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}

.zoom {
  transition: transform .2s; /* Animation */
  <!-- width: 200px;
  height: 200px; -->
}

.zoom:hover {
  transform: scale(3.5); /* (150% zoom - Note: if the zoom is too large, it will go outside of the viewport) */
}

.container { position:relative; }
.container video {
    position:relative;
    z-index:0;
}
.overlay {
    position:absolute;
    top:0;
    left:0;
    z-index:1;
}
span {
    font-style: inherit;
    font-weight: inherit;
}
.icon {
    align-items: center;
    display: inline-flex;
    justify-content: center;
    height: 1rem;
    width: 1rem;
}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">



<div id="primarycontent">
<center><h1><strong>VIOLA: Imitation Learning for Vision-Based Manipulation <br> with Object Proposals Priors</strong></h1></center>
<center><h2>
    <a href="https://zhuyifengzju.github.io/">Yifeng Zhu<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
    <a href="">Abhishek Joshi<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
<a href="https://cs.utexas.edu/~pstone">Peter Stone<sup>1, 2</sup></a>&nbsp;&nbsp;&nbsp; 
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu<sup>1</sup></a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/"><sup>1</sup>The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;   
        <a href="https://www.cs.utexas.edu/"><sup>2</sup>Sony AI</a>&nbsp;&nbsp;&nbsp;   		
    </h2></center>

	<center><h2><a href="https://arxiv.org/abs/2210.11339"><span class="icon">
                      <svg class="svg-inline--fa fa-file-pdf fa-w-12" aria-hidden="true" focusable="false" data-prefix="fas" data-icon="file-pdf" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512" data-fa-i2svg=""><path fill="currentColor" d="M181.9 256.1c-5-16-4.9-46.9-2-46.9 8.4 0 7.6 36.9 2 46.9zm-1.7 47.2c-7.7 20.2-17.3 43.3-28.4 62.7 18.3-7 39-17.2 62.9-21.9-12.7-9.6-24.9-23.4-34.5-40.8zM86.1 428.1c0 .8 13.2-5.4 34.9-40.2-6.7 6.3-29.1 24.5-34.9 40.2zM248 160h136v328c0 13.3-10.7 24-24 24H24c-13.3 0-24-10.7-24-24V24C0 10.7 10.7 0 24 0h200v136c0 13.2 10.8 24 24 24zm-8 171.8c-20-12.2-33.3-29-42.7-53.8 4.5-18.5 11.6-46.6 6.2-64.2-4.7-29.4-42.4-26.5-47.8-6.8-5 18.3-.4 44.1 8.1 77-11.6 27.6-28.7 64.6-40.8 85.8-.1 0-.1.1-.2.1-27.1 13.9-73.6 44.5-54.5 68 5.6 6.9 16 10 21.5 10 17.9 0 35.7-18 61.1-61.8 25.8-8.5 54.1-19.1 79-23.2 21.7 11.8 47.1 19.5 64 19.5 29.2 0 31.2-32 19.7-43.4-13.9-13.6-54.3-9.7-73.6-7.2zM377 105L279 7c-4.5-4.5-10.6-7-17-7h-6v128h128v-6.1c0-6.3-2.5-12.4-7-16.9zm-74.1 255.3c4.1-2.7-2.5-11.9-42.8-9 37.1 15.8 42.8 9 42.8 9z"></path></svg><!-- <i class="fas fa-file-pdf"></i> Font Awesome fontawesome.com -->
                  </span> Paper</a> | 
                  <a href=""><span class="icon" style="height: 1.7rem;width: 1.3rem;">
                    <svg class="svg-inline--fa fa-youtube fa-w-18" aria-hidden="true" focusable="false" data-prefix="fab" data-icon="youtube" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 576 512" data-fa-i2svg=""><path fill="currentColor" d="M549.655 124.083c-6.281-23.65-24.787-42.276-48.284-48.597C458.781 64 288 64 288 64S117.22 64 74.629 75.486c-23.497 6.322-42.003 24.947-48.284 48.597-11.412 42.867-11.412 132.305-11.412 132.305s0 89.438 11.412 132.305c6.281 23.65 24.787 41.5 48.284 47.821C117.22 448 288 448 288 448s170.78 0 213.371-11.486c23.497-6.321 42.003-24.171 48.284-47.821 11.412-42.867 11.412-132.305 11.412-132.305s0-89.438-11.412-132.305zm-317.51 213.508V175.185l142.739 81.205-142.739 81.201z"></path></svg><!-- <i class="fab fa-youtube"></i> Font Awesome fontawesome.com -->
                </span> Video</a> |
                 <a href="https://github.com/UT-Austin-RPL/VIOLA"><span class="icon" style="height: 1.3rem;width: 1.3rem;">
                    <svg class="svg-inline--fa fa-github fa-w-16" aria-hidden="true" focusable="false" data-prefix="fab" data-icon="github" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512" data-fa-i2svg=""><path fill="currentColor" d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3.3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5.3-6.2 2.3zm44.2-1.7c-2.9.7-4.9 2.6-4.6 4.9.3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3.7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3.3 2.9 2.3 3.9 1.6 1 3.6.7 4.3-.7.7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3.7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3.7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"></path></svg><!-- <i class="fab fa-github"></i> Font Awesome fontawesome.com -->
                </span> Code</a> | 
                <a href="./src/bib.txt"><span class="icon" style="height: 1.5rem;width: 1.5rem;">
                    <svg class="svg-inline--fa fa-bibtex fa-w-16" aria-hidden="true" focusable="false" data-prefix="fab" data-icon="bibtex" role="img" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120" data-fa-i2svg=""><path fill="currentColor" d="m 29.09375,11.234375 c -3.183804,0 -5.71875,2.566196 -5.71875,5.75 l 0,94.031255 c 0,3.1838 2.534946,5.75 5.71875,5.75 l 69.8125,0 c 3.1838,0 5.71875,-2.5662 5.71875,-5.75 l 0,-70.656255 -21.03125,0 c -4.306108,0 -8.0625,-3.141109 -8.0625,-7.3125 l 0,-21.8125 -46.4375,0 z m 50.4375,0 0,21.8125 c 0,1.714122 1.631968,3.3125 4.0625,3.3125 l 21.03125,0 -25.09375,-25.125 z m -46.1875,51.3125 19.03125,0 0.25,5.46875 -0.625,0 c -0.126107,-0.962831 -0.313482,-1.64983 -0.53125,-2.0625 -0.355356,-0.664804 -0.841468,-1.159242 -1.4375,-1.46875 -0.584605,-0.320929 -1.349667,-0.499979 -2.3125,-0.5 l -3.28125,0 0,17.84375 c -1.2e-5,1.432815 0.15925,2.300914 0.46875,2.65625 0.435561,0.481426 1.094449,0.718751 2,0.71875 l 0.8125,0 0,0.625 -9.875,0 0,-0.625 0.8125,0 c 0.985768,10e-7 1.712342,-0.278949 2.125,-0.875 0.252166,-0.366798 0.343741,-1.193273 0.34375,-2.5 l 0,-17.84375 -2.8125,0 c -1.088943,2.1e-5 -1.854004,0.08955 -2.3125,0.25 -0.596054,0.217809 -1.107139,0.631046 -1.53125,1.25 -0.424114,0.618995 -0.66977,1.476719 -0.75,2.53125 l -0.65625,0 0.28125,-5.46875 z m 37.3125,0 10.78125,0 0,0.625 c -0.91701,0.03441 -1.562385,0.173884 -1.90625,0.4375 -0.332422,0.263659 -0.500009,0.554071 -0.5,0.875 -9e-6,0.424133 0.293541,1.061183 0.84375,1.875 l 3.5625,5.34375 4.15625,-5.25 c 0.481406,-0.618955 0.771818,-1.051979 0.875,-1.28125 0.11461,-0.229229 0.187481,-0.446767 0.1875,-0.6875 -1.9e-5,-0.240691 -0.112469,-0.472828 -0.25,-0.65625 -0.171956,-0.24069 -0.361381,-0.40828 -0.625,-0.5 -0.263655,-0.10314 -0.830966,-0.14476 -1.65625,-0.15625 l 0,-0.625 8.28125,0 0,0.625 c -0.653386,0.03441 -1.181122,0.140585 -1.59375,0.3125 -0.618997,0.263659 -1.171709,0.615484 -1.6875,1.0625 -0.515833,0.447058 -1.278845,1.265207 -2.21875,2.46875 l -4.625,5.90625 5.03125,7.46875 c 1.386942,2.063254 2.397654,3.387302 3.0625,3.9375 0.676265,0.538738 1.530851,0.81769 2.5625,0.875 l 0,0.625 -10,0 0,-0.625 c 0.66481,-0.01146 1.147784,-0.06141 1.46875,-0.1875 0.240697,-0.103161 0.44472,-0.262423 0.59375,-0.46875 0.16046,-0.217786 0.249982,-0.438461 0.25,-0.65625 -1.8e-5,-0.263636 -0.05311,-0.54886 -0.15625,-0.8125 -0.08025,-0.19486 -0.418566,-0.686159 -0.96875,-1.5 l -3.9375,-5.96875 -4.875,6.25 c -0.515819,0.664828 -0.834344,1.114502 -0.9375,1.34375 -0.10316,0.217789 -0.156256,0.44679 -0.15625,0.6875 -6e-6,0.366801 0.159256,0.665539 0.46875,0.90625 0.30948,0.240713 0.910092,0.37186 1.78125,0.40625 l 0,0.625 -8.28125,0 0,-0.625 c 0.584586,-0.05731 1.075886,-0.160349 1.5,-0.34375 0.710673,-0.298024 1.389347,-0.714398 2.03125,-1.21875 0.641896,-0.504347 1.393444,-1.26941 2.21875,-2.3125 l 5.5,-6.9375 -4.59375,-6.75 c -1.249419,-1.822518 -2.316354,-3.000816 -3.1875,-3.5625 -0.871152,-0.573103 -1.865215,-0.87184 -3,-0.90625 l 0,-0.625 z m -19.3125,7.34375 17.96875,0 0.25,5.09375 -0.6875,0 c -0.240731,-1.226469 -0.514493,-2.07273 -0.8125,-2.53125 -0.28658,-0.458478 -0.708141,-0.821767 -1.28125,-1.0625 -0.458515,-0.17192 -1.279802,-0.249978 -2.4375,-0.25 l -6.375,0 0,9.21875 5.125,0 c 1.329636,1.3e-5 2.209198,-0.192549 2.65625,-0.59375 0.596035,-0.52726 0.93121,-1.451586 1,-2.78125 l 0.625,0 0,8.125 -0.625,0 c -0.160491,-1.134778 -0.30829,-1.897791 -0.46875,-2.21875 -0.206341,-0.401177 -0.561302,-0.708239 -1.03125,-0.9375 -0.469976,-0.229239 -1.181951,-0.343739 -2.15625,-0.34375 l -5.125,0 0,7.6875 c -7e-6,1.031628 0.0333,1.677002 0.125,1.90625 0.09169,0.217789 0.239493,0.393702 0.46875,0.53125 0.229242,0.12609 0.701842,0.187501 1.34375,0.1875 l 3.9375,0 c 1.318173,10e-7 2.278935,-0.09785 2.875,-0.28125 0.596034,-0.183399 1.137283,-0.55501 1.6875,-1.09375 0.710657,-0.710672 1.473668,-1.754683 2.21875,-3.1875 l 0.6875,0 -2,5.8125 -17.96875,0 0,-0.625 0.8125,0 c 0.550198,0 1.069611,-0.111362 1.5625,-0.375 0.366797,-0.183395 0.592659,-0.476948 0.71875,-0.84375 0.13755,-0.366798 0.218745,-1.11521 0.21875,-2.25 l 0,-15.15625 c -5e-6,-1.478642 -0.139479,-2.374854 -0.4375,-2.71875 -0.412653,-0.458478 -1.099652,-0.687478 -2.0625,-0.6875 l -0.8125,0 0,-0.625 z"></path></svg><!-- <i class="fab fa-bibtex"></i> Font Awesome fontawesome.com -->
                </span> Bibtex</a> </h2></center>

 <center><p><span style="font-size:20px;">6th Conference on Robot Learning, Auckland, New Zealand</span></p></center>
<!-- <p> -->
<!--   </p><table border="0" cellspacing="10" cellpadding="0" align="center">  -->
<!--   <tbody> -->
<!--   <tr> -->
<!--   <\!-- For autoplay -\-> -->
<!-- <iframe width="560" height="315" -->
<!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4?autoplay=1&mute=1&loop=1" -->
<!--   autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -->
<!--   <\!-- No autoplay -\-> -->
<!-- <\!-- <iframe width="560" height="315" -\-> -->
<!-- <\!--   src="https://www.youtube.com/embed/GCfs3DJ4aO4" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   -\-> -->

<!-- </tr> -->
<!-- </tbody> -->
<!-- </table> -->


<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody><tr>  <td align="center" valign="middle">
  <!-- <a href="./src/approach.gif"> <img src="./src/approach.gif" style="width:80%;">  
  </a> -->
    <!-- <a href="./videos/VIOLA supp approach.mp4"> <video src="./videos/VIOLA supp approach.mp4" type="video/mp4" style="width:80%;"/>   -->
     <video muted="" autoplay="" loop="" width="88%">
        <source src="videos/teaser no fadeout.mp4" type="video/mp4">
      </video>
  <!-- </a> -->
  </td>
  </tr>

</tbody>
</table>

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">

We introduce VIOLA, an object-centric imitation learning approach to learning closed-loop visuomotor policies for robot manipulation. Our approach constructs object-centric representations based on general object proposals from a pre-trained vision model. It uses a transformer-based policy to reason over these representations and attends to the task-relevant visual factors for action prediction. Such object-based structural priors improve deep imitation learning algorithm's robustness against object variations and environmental perturbations. We quantitatively evaluate VIOLA in simulation and on real robots. VIOLA outperforms the state-of-the-art imitation learning methods by 45.8% in success rates. It has also been deployed successfully on a physical robot to solve challenging long-horizon tasks, such as dining table arrangements and coffee making.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">Method Overview</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody><tr>  <td align="center" valign="middle">
  <!-- <a href="./src/approach.gif"> <img src="./src/approach.gif" style="width:80%;">  
  </a> -->
    <!-- <a href="./videos/VIOLA supp approach.mp4"> <video src="./videos/VIOLA supp approach.mp4" type="video/mp4" style="width:80%;"/>   -->
     <video controls="" muted="" autoplay="" loop="" width="88%">
        <source src="videos/VIOLA supp approach.mp4" type="video/mp4">
      </video>
  <!-- </a> -->
  </td>
  </tr>

</tbody>
</table>
  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Overview of VIOLA. We use a pre-trained RPN to get general object proposals that allow us to learn object-centric visuomotor skills.
</p></td></tr></table>

  
<!-- <br><br><hr> <h1 align="center">Hierarchical Policy Model</h1>  <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/Hierarchical-Policy.png"> <img
src="./src/Hierarchical-Policy.png" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table width=800px><tr><td> <p align="justify" width="20%">Overview of
the hierarchical policy. Given a workspace observation, the meta
controller selects the skill index and generates the latent subgoal
vector. Then the selected sensorimotor skill generates action
conditioned on observed images, proprioception, and the subgoal
vector.  </p></td></tr></table>
<br> -->



<br><hr>
<h1 align="center">Real Robot Experiment</h1>
<!-- <table border="0" cellspacing="10"
cellpadding="0">
<tr>
<td>
<p> </p>
</td>
</tr>
</table> -->


<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
  <td align="center">
  <p align="justify" width="20%">Our evaluation on real-robot tasks is shown in the following table. We show that VIOLA learns the manipulation policies with behavioral cloning algorithms much better than the state-of-the-art baseline, BC-RNN. Notably, in the <tt>Make-Coffee</tt> task, the baseline fails to complete the task in any attempt, while VIOLA is able to achieve 60&#x25;. This empirical result further proves the effectiveness of VIOLA.</p>
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
  <td align="center">
<svg
   xmlns:dc="http://purl.org/dc/elements/1.1/"
   xmlns:cc="http://creativecommons.org/ns#"
   xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
   xmlns:svg="http://www.w3.org/2000/svg"
   xmlns="http://www.w3.org/2000/svg"
   viewBox="0 0 960 288"
   height="288"
   width="960"
   xml:space="preserve"
   id="svg2"
   version="1.1"><metadata
     id="metadata8"><rdf:RDF><cc:Work
         rdf:about=""><dc:format>image/svg+xml</dc:format><dc:type
           rdf:resource="http://purl.org/dc/dcmitype/StillImage" /></cc:Work></rdf:RDF></metadata><defs
     id="defs6"><clipPath
       id="clipPath20"
       clipPathUnits="userSpaceOnUse"><path
         id="path18"
         d="M 0,0.2400048 H 720 V 216 H 0 Z" /></clipPath></defs><g
     transform="matrix(1.3333333,0,0,-1.3333333,0,288)"
     id="g10"><g
       id="g12" /><g
       id="g14"><g
         clip-path="url(#clipPath20)"
         id="g16"><path
           id="path22"
           style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
           d="M 0,216 H 720 V 0.2400048 H 0 Z" /></g></g><g
       id="g24"><path
         id="path26"
         style="fill:#ffffff;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="M 0,216 H 720 V 5.632639e-6 H 0 Z" /><path
         id="path28"
         style="fill:#ed7d31;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="m 130.5003,170.4375 h 64.1251 V 41.06245 h -64.1251 z" /><path
         id="path30"
         style="fill:#ed7d31;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="m 379.8838,139.5 h 64.125 V 41.06245 h -64.125 z" /><path
         id="path32"
         style="fill:#ed7d31;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="m 629.2672,139.5 h 64.1251 V 41.06245 h -64.1251 z" /><path
         id="path34"
         style="fill:#4472c4;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="M 48.00031,105.7502 H 112.1254 V 40.95017 H 48.00031 Z" /><path
         id="path36"
         style="fill:#4472c4;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="m 297.3838,74.25 h 64.125 V 40.95016 h -64.125 z" /><g
         transform="matrix(0.24,0,0,0.24,56.68779,-100.8)"
         id="g38"><text
           id="text42"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan40"
             y="0"
             x="0 55.73 111.46 139.09">36.7</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,139.1878,-31.19999)"
         id="g44"><text
           id="text48"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan46"
             y="0"
             x="0 55.73 111.46 139.09">76.7</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,306.0712,-129.84)"
         id="g50"><text
           id="text54"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan52"
             y="0"
             x="0 55.73 111.46 139.09">20.0</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,389.6962,-60.23999)"
         id="g56"><text
           id="text60"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan58"
             y="0"
             x="0 55.73 111.46 139.09">60.0</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,640.2046,-60.23999)"
         id="g62"><text
           id="text66"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan64"
             y="0"
             x="0 55.73 111.46 139.09">60.0</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,562.1422,-161.04)"
         id="g68"><text
           id="text72"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan70"
             y="0"
             x="0 55.73 83.360001">0.0</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,41.68779,-201.36)"
         id="g74"><text
           id="text78"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan76"
             y="0"
             x="0 72.400002 94.800003 150.5 172.89999 228.60001">Dining</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,111.1878,-201.36)"
         id="g80"><text
           id="text84"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan82"
             y="0"
             x="0">-</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,119.1878,-201.36)"
         id="g86"><text
           id="text90"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan88"
             y="0"
             x="0 60.939999 83.379997 139.12 166.75999 222.5 278.23999 333.98001 367.32001">PlateFork</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,306.3188,-201.36)"
         id="g92"><text
           id="text96"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan94"
             y="0"
             x="0 72.400002 94.800003 150.5 172.89999 228.60001">Dining</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,375.8188,-201.36)"
         id="g98"><text
           id="text102"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan100"
             y="0"
             x="0">-</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,383.8188,-201.36)"
         id="g104"><text
           id="text108"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan106"
             y="0"
             x="0 66.669998 122.44 194.81">Bowl</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,560.5172,-201.36)"
         id="g110"><text
           id="text114"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan112"
             y="0"
             x="0 83.330002 139.06 189.06">Make</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,619.2672,-201.36)"
         id="g116"><text
           id="text120"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan118"
             y="0"
             x="0">-</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,627.2672,-201.36)"
         id="g122"><text
           id="text126"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan124"
             y="0"
             x="0 72.400002 128.10001 155.7 183.3 239">Coffee</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,512.0364,-23.99999)"
         id="g128"><text
           id="text132"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan130"
             y="0"
             x="0 66.669998">BC</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,545.4114,-23.99999)"
         id="g134"><text
           id="text138"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan136"
             y="0"
             x="0">-</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,553.4114,-23.99999)"
         id="g140"><text
           id="text144"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan142"
             y="0"
             x="0 66.669998 139.03999">RNN</tspan></text>
</g><g
         transform="matrix(0.24,0,0,0.24,646.1424,-23.99999)"
         id="g146"><text
           id="text150"
           style="font-variant:normal;font-weight:300;font-size:100px;font-family:Helvetica;-inkscape-font-specification:Helvetica-Light;writing-mode:lr-tb;fill:#000000;fill-opacity:1;fill-rule:nonzero;stroke:none"
           transform="matrix(1,0,0,-1,0,899)"><tspan
             id="tspan148"
             y="0"
             x="0 60.939999 88.580002 166.22 221.96001">VIOLA</tspan></text>
</g><path
         id="path152"
         style="fill:#ed7d31;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="m 619.6575,210.1011 h 21.6 v -21.6 h -21.6 z" /><path
         id="path154"
         style="fill:#4472c4;fill-opacity:1;fill-rule:nonzero;stroke:none"
         d="m 485.9646,210.1011 h 21.6 v -21.6 h -21.6 z" /></g></g></svg>
</td>
</tr>
</tbody>
</table>

<h2 align="center">Qualitative Real Robot Demo</h2>

<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> </p></td></tr></table>
  <table border="0" cellspacing="10" cellpadding="0"
  align="center">
  <tbody>
  <tr>
  <!-- For autoplay -->

<video width="450" height="253" controls="" muted="" autoplay="" loop="" frameborder="5">
  <source src="videos/dining two policies.mp4" type="video/mp4">
</video>

<video width="450" height="253" controls="" muted="" autoplay="" loop="" frameborder="5">
  <source src="videos/make two coffee.mp4" type="video/mp4">
</video>
</tr>
<tr>
 <td align="center" valign="middle">
 We can sequentially execute <tt>Dining-PlateFork</tt> and <tt>Dining-Bowl</tt> policies.
</td> 
<td align="center" valign="middle">
 This video shows that the learned policies making two coffees in a row.
</td>
</tr>
</tbody>
</table>
<br>

<h2 align="center">Our policies are robust to scenarios where unseen distracting objects are present</h2>
<h3 align="center">(The cup and the strawberry in bowl were never present in demonstrations)</h3>
  <table border="0" cellspacing="10" cellpadding="0"
  align="center">
  <tbody>
  <tr>
  <!-- For autoplay -->
<video width="450" height="253" controls="" muted="" autoplay="" loop="" frameborder="5">
  <source src="videos/dining distracting 1.mp4" type="video/mp4">
</video>

<video width="450" height="253" controls="" muted="" autoplay="" loop="" frameborder="5">
  <source src="videos/dining distracting 2.mp4" type="video/mp4">
</video>
</tr>
</tbody>
</table>
<br>


<h2 align="center">A no-cut video of <b>10</b> <tt>Make-Coffee</tt> rollouts</h2>

<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> </p></td></tr></table>
  <table border="0" cellspacing="10" cellpadding="0"
  align="center">
  <tbody>
  <tr>
<iframe width="750" height="421" src="https://www.youtube.com/embed/DVFSPSa7GsQ?autoplay=1&mute=1&loop=1&playlist=DVFSPSa7GsQ" autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</tr>
</tbody>
</table>

<br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> The authors would like to specially thank Yue Zhao for the great discussion on the project and the insightful feedback on the manuscript.
This work has taken place
in the Robot Perception and Learning Group (RPL) and Learning Agents
Research Group (LARG) at UT Austin. RPL research has been partially supported
by the National Science Foundation (CNS-1955523, FRR-2145283), the Office of Naval Research
(N00014-22-1-2204), and the Amazon Research Awards. LARG
research is supported in part by NSF (CPS-1739964, IIS-1724157,
NRI-1925082), ONR (N00014-18-2243), FLI (RFP2-000), ARO
(W911NF-19-2-0333), DARPA, Lockheed Martin, GM, and Bosch. Peter Stone
serves as the Executive Director of Sony AI America and receives
financial compensation for this work. The terms of this arrangement
have been reviewed and approved by the University of Texas at Austin
in accordance with its policy on objectivity in research.
				
<!-- The webpage template was borrowed from some <a href="https://nvlabs.github.io/SPADE/">GAN folks</a>. -->
</left></td></tr></table>
<br><br>

<div style="display:none">
<!-- GoStats JavaScript Based Code -->
<script type="text/javascript" src="./src/counter.js"></script>
<script type="text/javascript">_gos='c3.gostats.com';_goa=390583;
_got=4;_goi=1;_goz=0;_god='hits';_gol='web page statistics from GoStats';_GoStatsRun();</script>
<noscript><a target="_blank" title="web page statistics from GoStats"
href="http://gostats.com"><img alt="web page statistics from GoStats"
src="http://c3.gostats.com/bin/count/a_390583/t_4/i_1/z_0/show_hits/counter.png"
style="border-width:0" /></a></noscript>
</div>
<!-- End GoStats JavaScript Based Code -->
<!-- </center></div></body></div> -->

