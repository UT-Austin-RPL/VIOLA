---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
  <title>VIOLA: Object-Centric Imitation Learning for Vision-Based Robot Manipulation</title>


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
hr
  {
    border: 0;
    height: 1px;
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
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


</head>

<body data-gr-c-s-loaded="true">



<div id="primarycontent">
<center><h1><strong>VIOLA: Object-Centric Imitation Learning for Vision-Based Robot Manipulation</strong></h1></center>
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

	<center><h2><a href="">Paper</a> | <a href="">Video</a> | <a href="https://github.com/UT-Austin-RPL/VIOLA">Code</a> | <a href="./src/bib.txt">Bibtex</a> </h2></center>
    
 <center><p><span style="font-size:20px;">Technical Report</span></p></center>
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
  <tbody><tr>  <td align="center" valign="middle"><a href="./src/approach.gif"> <img src="./src/approach.gif" style="width:80%;">  </a></td>
  </tr>

</tbody>
</table>
  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  Overview of BUDS. We construct hierarchical task structures of demonstration sequences in a bottom-up manner, from which we obtain temporal segments for discovering and learning sensorimotor skills.
</p></td></tr></table>

  
<br><br><hr> <h1 align="center">Hierarchical Policy Model</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
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
<br>

<hr>


<h1 align="center">Simulation Experiment</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">We show a qualitative comparison
  between two baselines (vanilla BC and Changepoint Detction) and our
  method BUDS on
  <tt>Kitchen</tt>, achieving 24.4%, 23.4%, and 72.0% task success rate respectively. BUDS learns skills that lead to better execution,
  and we've shown the quantitative result in Table 1 of the paper.</p>
</td></tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr><p></p></tr>
<tr>
<td align="center" valign="middle">
<iframe width="260" height="148"
src="https://www.youtube.com/embed/XlYSLa75pvI?autoplay=1&mute=1&playlist=XlYSLa75pvI&loop=1"
autoplay="true" frameborder="0" allow="accelerometer; autoplay;
clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe></td>
<td align="center" valign="middle">
<iframe width="260" height="148" src="https://www.youtube.com/embed/rs50JBtXCIU?autoplay=1&mute=1&playlist=rs50JBtXCIU&loop=1"
autoplay="true" frameborder="0" allow="accelerometer; autoplay;
clipboard-write; encrypted-media; gyroscope; picture-in-picture"
allowfullscreen></iframe>
</td>
<td align="center" valign="middle">
<iframe width="260" height="148"  src="https://www.youtube.com/embed/cjKT8tpGk8A?autoplay=1&mute=1&playlist=cjKT8tpGk8A&loop=1" autoplay="true" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</td>
 </tr>
 <tr>
 <td align="center" valign="middle">
 BC Baseline [1]
</td>
 <td align="center" valign="middle">
 CP Baseline [2]
</td>
 <td align="center" valign="middle">
 BUDS (Ours)
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0"> 
<tr align="left">
<td> [1] Mandlekar et al. What Matters in Learning from Offline Human Demonstrations for Robot Manipulation</td> </tr>
<br> 
<tr align="left"> <td> [2] OREO</td> </tr>
</table>


<br><hr>
<h1 align="center">Real Robot Experiment</h1>
<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p> </p></td></tr></table>
  <table border="0" cellspacing="10" cellpadding="0"
  align="center">
  <tbody>
  <tr>
  <!-- For autoplay -->
<iframe width="450" height="253"  src="https://www.youtube.com/embed/nF5vHSxLSKM?autoplay=1&mute=1&playlist=nF5vHSxLSKM&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  
<iframe width="450" height="253"  src="https://www.youtube.com/embed/apneKhEp4zk?autoplay=1&mute=1&playlist=apneKhEp4zk&loop=1" autoplay="true" frameborder="5" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

</tr>
</tbody>
</table>
<br>

<br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> This work has taken place
in the Robot Perception and Learning Group (RPL) and Learning Agents
Research Group (LARG) at UT Austin.  RPL research has been partially
supported by NSF CNS-1955523, the MLL Research Award from the Machine
Learning Laboratory at UT-Austin, and the Amazon Research Awards. LARG
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

