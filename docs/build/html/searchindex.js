Search.setIndex({envversion:47,filenames:["AdaptiveIntegrator","Comparator","ControlMechanism","ControlSignal","DDM","DefaultControlMechanism","EVCMechanism","InputState","Learning","Log","Mapping","Mechanism","MonitoringMechanism","OutputState","ParameterState","Preferences","Process","ProcessingMechanism","Projection","Run","State","System","Transfer","Utilities","UtilityFunction","WeightedError","index"],objects:{"":{Comparator:[1,0,0,"-"],ControlMechanism:[2,0,0,"-"],ControlSignal:[3,0,0,"-"],DDM:[4,0,0,"-"],DefaultControlMechanism:[5,0,0,"-"],InputState:[7,0,0,"-"],Mapping:[10,0,0,"-"],Mechanism:[11,0,0,"-"],MonitoringMechanism:[12,0,0,"-"],OutputState:[13,0,0,"-"],ParameterState:[14,0,0,"-"],Preferences:[15,0,0,"-"],Process:[16,0,0,"-"],ProcessingMechanism:[17,0,0,"-"],Projection:[18,0,0,"-"],Run:[19,0,0,"-"],State:[20,0,0,"-"],System:[21,0,0,"-"],Transfer:[22,0,0,"-"],WeightedError:[25,0,0,"-"]},"Comparator.Comparator":{"__execute__":[1,1,1,""],terminate_function:[1,1,1,""]},"ControlMechanism.ControlMechanism_Base":{"__execute__":[2,1,1,""],Linear:[2,4,1,""],instantiate_control_mechanism_input_state:[2,1,1,""],instantiate_control_signal_projection:[2,1,1,""]},"ControlMechanism.ControlMechanism_Base.Linear":{"function":[2,1,1,""],derivative:[2,1,1,""]},"ControlSignal.ControlSignal":{compute_cost:[3,1,1,""],execute:[3,1,1,""]},"ControlSignal.ControlSignalChannel":{"__getnewargs__":[3,1,1,""],"__new__":[3,5,1,""],"__repr__":[3,1,1,""],inputState:[3,3,1,""],outputIndex:[3,3,1,""],outputState:[3,3,1,""],outputValue:[3,3,1,""],variableIndex:[3,3,1,""],variableValue:[3,3,1,""]},"ControlSignal.ControlSignalValuesTuple":{"__getnewargs__":[3,1,1,""],"__new__":[3,5,1,""],"__repr__":[3,1,1,""],cost:[3,3,1,""],intensity:[3,3,1,""]},"DDM.DDM":{"__execute__":[4,1,1,""],ou_update:[4,1,1,""],terminate_function:[4,1,1,""]},"DefaultControlMechanism.ControlSignalChannel":{"__getnewargs__":[5,1,1,""],"__new__":[5,5,1,""],"__repr__":[5,1,1,""],inputState:[5,3,1,""],outputIndex:[5,3,1,""],outputState:[5,3,1,""],outputValue:[5,3,1,""],variableIndex:[5,3,1,""],variableValue:[5,3,1,""]},"DefaultControlMechanism.DefaultControlMechanism":{instantiate_control_signal_channel:[5,1,1,""],instantiate_control_signal_projection:[5,1,1,""]},"Mapping.Mapping":{execute:[10,1,1,""]},"Mechanism.MechanismList":{"__getitem__":[11,1,1,""],mechanisms:[11,3,1,""],names:[11,3,1,""],outputStateNames:[11,3,1,""],outputStateValues:[11,3,1,""],values:[11,3,1,""]},"Mechanism.MechanismTuple":{"__getnewargs__":[11,1,1,""],"__new__":[11,5,1,""],"__repr__":[11,1,1,""],mechanism:[11,3,1,""],params:[11,3,1,""],phase:[11,3,1,""]},"Mechanism.Mechanism_Base":{execute:[11,1,1,""],function_params:[11,3,1,""],initialize:[11,1,1,""],inputState:[11,3,1,""],inputStates:[11,3,1,""],inputValue:[11,3,1,""],name:[11,3,1,""],outputState:[11,3,1,""],outputStates:[11,3,1,""],outputValue:[11,3,1,""],parameterStates:[11,3,1,""],phaseSpec:[11,3,1,""],prefs:[11,3,1,""],processes:[11,3,1,""],run:[11,1,1,""],systems:[11,3,1,""],timeScale:[11,3,1,""],value:[11,3,1,""],variable:[11,3,1,""]},"MonitoringMechanism.MonitoringMechanism_Base":{update_monitored_state_changed_attribute:[12,1,1,""]},"ParameterState.ParameterState":{update:[14,1,1,""]},"Process.ProcessList":{processNames:[16,3,1,""],processes:[16,3,1,""]},"Process.ProcessTuple":{"__getnewargs__":[16,1,1,""],"__new__":[16,5,1,""],"__repr__":[16,1,1,""],input:[16,3,1,""],process:[16,3,1,""]},"Process.Process_Base":{clamp_input:[16,3,1,""],execute:[16,1,1,""],input:[16,3,1,""],inputValue:[16,3,1,""],learning:[16,3,1,""],mechanismNames:[16,3,1,""],monitoringMechanisms:[16,3,1,""],name:[16,3,1,""],numPhases:[16,3,1,""],originMechanisms:[16,3,1,""],outputState:[16,3,1,""],pathway:[16,3,1,""],prefs:[16,3,1,""],processInputStates:[16,3,1,""],results:[16,3,1,""],run:[16,1,1,""],systems:[16,3,1,""],terminalMechanisms:[16,3,1,""],timeScale:[16,3,1,""],value:[16,3,1,""]},"Projection.Projection_Base":{name:[18,3,1,""],paramInstanceDefaults:[18,3,1,""],paramNames:[18,3,1,""],params:[18,3,1,""],paramsCurrent:[18,3,1,""],prefs:[18,3,1,""],receiver:[18,3,1,""],sender:[18,3,1,""],value:[18,3,1,""],variable:[18,3,1,""]},"State.State_Base":{check_projection_receiver:[20,1,1,""],check_projection_sender:[20,1,1,""],instantiate_projection_from_state:[20,1,1,""],instantiate_projections_to_state:[20,1,1,""],parse_projection_ref:[20,1,1,""],update:[20,1,1,""]},"System.System_Base":{InspectOptions:[21,4,1,""],controlMechanisms:[21,3,1,""],execute:[21,1,1,""],executionGraph:[21,3,1,""],executionList:[21,3,1,""],execution_graph_mechs:[21,3,1,""],execution_sets:[21,3,1,""],graph:[21,3,1,""],initial_values:[21,3,1,""],inputValue:[21,3,1,""],inspect:[21,1,1,""],mechanisms:[21,3,1,""],mechanismsDict:[21,3,1,""],monitoringMechanisms:[21,3,1,""],name:[21,3,1,""],numPhases:[21,3,1,""],originMechanisms:[21,3,1,""],processes:[21,3,1,""],results:[21,3,1,""],run:[21,1,1,""],show:[21,1,1,""],terminalMechanisms:[21,3,1,""],value:[21,3,1,""]},"Transfer.Transfer":{"__execute__":[22,1,1,""],"function":[22,3,1,""],name:[22,3,1,""],prefs:[22,3,1,""],value:[22,3,1,""],variable:[22,3,1,""]},"WeightedError.WeightedError":{"__execute__":[25,1,1,""]},Comparator:{Comparator:[1,4,1,""],random:[1,2,1,""]},ControlMechanism:{ControlMechanism_Base:[2,4,1,""],random:[2,2,1,""]},ControlSignal:{ControlSignal:[3,4,1,""],ControlSignalChannel:[3,4,1,""],ControlSignalValuesTuple:[3,4,1,""],random:[3,2,1,""]},DDM:{DDM:[4,4,1,""],random:[4,2,1,""]},DefaultControlMechanism:{ControlSignalChannel:[5,4,1,""],DefaultControlMechanism:[5,4,1,""],random:[5,2,1,""]},InputState:{InputState:[7,4,1,""],random:[7,2,1,""]},Mapping:{Mapping:[10,4,1,""],random:[10,2,1,""]},Mechanism:{MechanismList:[11,4,1,""],MechanismTuple:[11,4,1,""],Mechanism_Base:[11,4,1,""],mechanism:[11,2,1,""],random:[11,2,1,""]},MonitoringMechanism:{MonitoringMechanism_Base:[12,4,1,""],random:[12,2,1,""]},OutputState:{OutputState:[13,4,1,""],instantiate_output_states:[13,2,1,""],random:[13,2,1,""]},ParameterState:{ParameterState:[14,4,1,""],instantiate_parameter_states:[14,2,1,""],random:[14,2,1,""]},Process:{ProcessInputState:[16,4,1,""],ProcessList:[16,4,1,""],ProcessTuple:[16,4,1,""],Process_Base:[16,4,1,""],process:[16,2,1,""],random:[16,2,1,""]},ProcessingMechanism:{ProcessingMechanism_Base:[17,4,1,""],random:[17,2,1,""]},Projection:{Projection_Base:[18,4,1,""],is_projection_spec:[18,2,1,""],random:[18,2,1,""]},Run:{random:[19,2,1,""],run:[19,2,1,""]},State:{State_Base:[20,4,1,""],check_parameter_state_value:[20,2,1,""],check_state_ownership:[20,2,1,""],instantiate_state:[20,2,1,""],instantiate_state_list:[20,2,1,""],random:[20,2,1,""]},System:{System_Base:[21,4,1,""],random:[21,2,1,""],system:[21,2,1,""]},Transfer:{Transfer:[22,4,1,""],random:[22,2,1,""]},WeightedError:{WeightedError:[25,4,1,""],random:[25,2,1,""]}},objnames:{"0":["py","module","Python module"],"1":["py","method","Python method"],"2":["py","function","Python function"],"3":["py","attribute","Python attribute"],"4":["py","class","Python class"],"5":["py","staticmethod","Python static method"]},objtypes:{"0":"py:module","1":"py:method","2":"py:function","3":"py:attribute","4":"py:class","5":"py:staticmethod"},terms:{"0x1058d0ef0":[],"0x1058ef438":[],"0x1060a8eb8":[],"0x1063284e0":1,"0x1068dce10":[],"0x106923eb8":[],"0x10a64dea0":[],"0x10ad490d0":[],"0x10ad84950":[],"0x10ae30f28":[1,2,3,4,5,7,10,13,14,16,21,22,25],"0x10ae36730":2,"0x10af360d0":[],"0x10af71950":[],"0x10b06f6a8":[],"0x10b64a0d0":[],"0x10b6670d0":[],"0x10b686950":[],"0x10b6a4950":[],"0x10b6e8dd8":[],"0x10b6e8eb8":[],"0x10bd40828":[],"0x10bd40e48":[],"0x10be99908":[],"0x10be99940":[],"0x10be99978":[],"0x10be999e8":[],"0x10c55bb70":[],"0x10c55be10":[],"0x10c695a58":[],"0x10c695a90":[],"0x10c695ac8":[],"0x10c695b38":[],"0x10c698898":[],"0x10c6988d0":[],"0x10c698908":[],"0x10c698978":[],"0x10c6b8c88":[],"0x10c6b8e80":[],"0x10c6b8ef0":[],"0x10c6b8f98":[],"0x10c723908":[],"0x10c723940":[],"0x10c723978":[],"0x10c7239e8":[],"0x10c875320":[],"0x10c875470":[],"0x10c875a58":16,"0x10c875cf8":16,"0x10c875d30":16,"0x10c875ef0":16,"0x10c899048":[],"0x10c899080":[],"0x10c8990f0":[],"0x10c899128":[],"0x10c899668":[],"0x10c899748":[],"0x10c8a20f0":[],"0x10c8a29b0":[],"0x10c8a69e8":[],"0x10c8a6a58":[],"0x10c8a6b00":[],"0x10c8a6b70":[],"0x10c8a6c18":[],"0x10c8a6d68":[],"0x10c8a6dd8":[],"0x10c8a6eb8":[],"0x10c8aa0b8":[],"0x10c8aa160":[],"0x10c8aa4a8":[],"0x10c8aa518":[],"0x10c8aa550":[],"0x10c8aa5c0":[],"0x10c8aa780":[],"0x10c8aa898":[],"0x10c8aaba8":[],"0x10c8b0c50":[],"0x10c8b4860":[],"0x10c924e80":21,"0x10c924fd0":21,"0x10c9620b8":[],"0x10c962240":[],"0x10c962278":[],"0x10c962320":[],"0x10c962390":[],"0x10c962780":[],"0x10c962b00":[],"0x10c966eb8":[],"0x10c966f28":[],"0x10cb73780":[],"0x10cb739b0":[],"0x10cbe6080":19,"0x10cbe62e8":19,"0x10cbe63c8":19,"0x10cbe6400":19,"0x10cbe6438":19,"0x10cbe66d8":19,"0x10cbe6748":19,"0x10cbe6860":19,"0x10cbe6a20":19,"0x10cc242e8":[],"0x10cc24438":[],"0x10cc246d8":[],"0x10cc24748":[],"0x10cc24780":[],"0x10cc247f0":[],"0x10cc24898":[],"0x10cc249b0":[],"0x10cc24e48":[],"1st":22,"2afc":4,"2nd":22,"3rd":22,"__execute__":[1,2,4,11,22,25],"__getitem__":11,"__getnewargs__":[3,5,11,16],"__init__":[1,2,3,4,7,10,11,13,14,16,18,20,21,22,25],"__new__":[3,5,11,16],"__repr__":[3,5,11,16],"_all_mech_tupl":[],"_allmechan":[],"_cl":[3,5,11,16],"_control_mech_tupl":21,"_instantiate_attributes_after_funct":2,"_instantiate_attributes_before_funct":2,"_instantiate_funct":[1,4,7,14],"_instantiate_graph":16,"_instantiate_receiv":10,"_instantiate_send":10,"_learning_mech_tupl":[],"_monitoring_mech_tupl":21,"_origin_mech_tupl":[],"_phasespecmax":16,"_processlist":[],"_stateregistri":20,"_terminal_mech_tupl":[],"_validate_param":[2,7,13,14,25],"_validate_vari":[7,10,13,14],"abstract":[2,11,12,16,17,18,20,21],"case":[3,4,7,10,11,13,16,19,20,21],"class":[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],"default":[],"enum":[1,2,3,21],"final":[16,21],"float":[4,11,22],"function":[],"int":[2,11,16,21],"long":[7,10,11,13,14,22],"new":[2,3,5,11,16,20],"return":[1,2,3,4,5,11,12,13,14,16,18,19,20,21,22,25],"static":[3,5,11,16],"super":[7,10,14],"true":[2,7,10,11,14,16,18,19,21],"while":[11,19,20,21],abil:[7,13],about:[5,11,16],abov:[1,2,4,10,14,16,18,19,20,21,25],absent:[7,11,13,20],accept:7,access:[11,16,20],accommod:2,accomod:2,accordingli:[11,12],accur:4,achiev:19,across:[19,20,22],activ:22,acycl:21,add:[2,5,10],addit:[1,4,11,13,18,20,21,22,25],adjac:16,adjust:[1,3,4,7,10,13,14,25],adjustment_cost:3,adjustmentcost:3,adjustmentcostfunct:3,affect:[19,21],after:[11,16,19,21],again:16,against:16,aggreg:[11,14,19],aggregr:11,algorithm:16,alia:[3,5,11,16],all:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],all_output:21,all_output_label:21,all_output_st:[2,21],alloc:[3,18],allocation_sampl:3,allocationpolici:2,allocationsampl:3,allow:[3,10,14,19,20,22],alon:[2,16,19],along:[3,16],also:[1,4,7,11,13,14,16,19,21,22,25],altern:[4,16],although:[7,14],alwai:[11,16,19],ambigu:16,among:[16,19,21],analysi:[4,19,21],analyt:[4,19],ani:[1,2,4,7,11,13,14,16,18,19,20,21,22,25],anoth:[7,10,11,14,16,21],anyth:14,appear:[2,16,19,21],append:19,appli:[2,16,21,22],appropri:[11,16,18,19,21],approxim:19,arg:[1,3,4,10,16,18,20,25],argument:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],arithmet:[7,14],arrai:[1,2,3,4,10,11,16,19,20,21,22,25],arri:2,assign:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],assign_default:[1,4,25],assign_funct:3,associ:[2,3,14,16,21,25],assum:[7,13,20],attent:4,attribut:[1,2,3,4,5,7,10,11,12,13,14,16,18,19,20,21,22,25],augment:[5,10],auto:3,autoassignmatrix:16,autom:3,automat:[1,3,7,10,13,14,16,19],autonumb:2,avail:16,averag:22,axi:[11,16,19,21],backpropag:16,bai:4,ballist:19,base:[1,2,3,4,10,11,16,18,19,20,21,25],basevalu:[11,14,20],becaus:19,been:[11,13,16,19,21],befor:[11,16,19,21,22],begin:21,behavior:[3,16],belong:[2,7,11,13,16,20,21],below:[3,4,11,16,19,21,22],best:19,between:[4,16,19,21],bia:[4,11,22],blue:11,bogacz:4,bogaczet:4,bool:[16,18,19,21],both:[7,10,13,16,19,20,21],branch:16,brown:[4,11],calcul:[2,4,7,16,19,25],call:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],call_after_execut:11,call_after_time_step:[16,19,21],call_after_tri:[16,19,21],call_before_execut:11,call_before_time_step:[16,19,21],call_before_tri:[16,19,21],caller:20,can:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],cannot:[11,16],cap:22,carri:11,categori:[1,3,4,7,10,13,14,20,21,25],centralclock:[4,16,19,21],chain:21,chang:[1,3,4,7,10,12,13,14,16,25],check:[10,14,16,20],check_parameter_state_valu:20,check_projection_receiv:20,check_projection_send:20,check_state_ownership:20,choic:4,clamp:16,clamp_input:16,clariti:[11,14,19],classnam:[10,20],classprefer:[1,3,4,10,11,14,16,18,20,21,22,25],classpreferencelevel:[1,3,4,10,14,20,25],close:[19,21],closest:19,cohen:4,collect:21,color:3,column:25,combin:[1,2,3,4,7,11,14,16,19,20,25],combinationoper:10,commit:19,common:19,commonli:[7,13],comparator_default_starting_point:1,comparator_preferenceset:1,comparator_sampl:1,comparator_target:1,comparatormechan:19,comparatoroutput:1,comparatorsampl:1,comparatortarget:1,comparis:1,comparison:[1,25],comparison_arrai:1,comparison_mean:1,comparison_oper:1,comparison_sum:1,comparison_sum_squar:1,comparison_typ:1,comparisonfunct:1,compat:[7,11,13,14,16,20,21],compati:[16,20],complet:13,compon:[4,11,16],comput:[1,2,3,4,20,25],compute_cost:3,concaten:11,concept:19,condit:[19,20],confid:[1,4],configur:16,confirm:[4,7,10,14,22],conform:[7,14],connect:[16,18,21],connectionist:16,consist:[11,16,19],constant:[18,20,22],constrain:21,constraint:[5,14,20],constraint_valu:20,constraint_value_nam:20,construct:[10,16,20,21],contain:[4,7,11,13,14,16,18,19,20,21],context:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],contextu:16,continu:[16,19],contribut:[3,25],control_function_typ:3,control_mechan:21,control_projection_receiv:21,control_sign:14,control_signal_param:11,controlmechanism_bas:2,controlmodulatedparamvalu:[1,4],controlproject:21,controlsign:[1,2,3,4,5,11,14,18,25],controlsignalajdustmentcostfunct:3,controlsignalchannel:[3,5],controlsignalcost:[2,3],controlsignaldurationcostfunct:3,controlsignalintensitycostfunct:3,controlsignallog:3,controlsignalpreferenceset:3,controlsignaltotalcostfunct:3,controlsignalvaluestupl:3,controlst:18,convei:[10,16,18],conveni:14,convent:[11,16,18,19,21,22],convert:[3,4,11,16,19],coordin:16,copi:[1,3,4,5,10,11,14,16,20,25],core:11,correct:[3,4,10,19],corresond:10,correspond:[1,4,7,11,13,16,19,20,21,22,25],corrrespond:21,cost:[2,3,5],costfunctionnam:3,count:[1,3,4,7,10,13,14,20,25],creat:[],criterion:19,curent:3,currenlti:16,current:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],current_monitored_st:12,currentstatetupl:[1,4],custom:[11,22],cycl:21,cyclic:21,ddm_analyt:4,ddm_default:4,ddm_distr:4,ddm_preferenceset:4,ddm_rt:4,ddm_updat:4,deal:16,debug:11,decai:4,decion:4,decis:4,decision_vari:4,defaul:2,default_alloc:3,default_allocation_sampl:3,default_input_valu:[2,4,5,11,16,19,21,22],default_matrix:10,default_projection_matrix:16,default_sample_and_target:1,default_sample_valu:3,defaultcontrol:[2,3,5,21],defaultcontrolalloc:[3,5],defaultcontrolmechan:[5,21],defaultmechan:[11,16],defaultprocess:21,defaultprocessingmechanism_bas:14,defaultreceiv:10,defaultsend:10,defin:[2,11,13,16,18,19,21,22],definit:[14,18],delet:[1,2,4],depend:[19,21],deriv:[2,25],describ:[4,11,16,21],descript:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],design:[3,11,16,19,21],desir:[11,16],destin:10,detail:[10,11,13,14,16,18,19,20,21,22],determin:[1,2,3,4,11,16,18,19,21,22],deviat:22,devoid:21,diciontari:21,dict:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],dict_output:21,dictionari:[1,3,4,7,10,11,13,14,16,19,20,21,22,25],dictoinari:11,differ:[1,4,11,16,18,20,21,25],differec:1,diffus:4,dimens:19,dimension:19,direct:[3,11,16,18,20,21],directi:19,directli:[1,3,4,7,10,11,13,14,16,19,22,25],disabl:[3,16,19,21],discuss:[19,21],distribut:4,divis:1,document:[],doe:[5,16,21],don:21,done:[7,11,13,19],drift:[4,11],drift_rat:4,dtype:22,duplic:[11,16,18,21,22],durat:[1,3,4],durationcost:3,durationcostfunct:3,dure:[11,16,19,21],each:[2,3,4,5,7,11,13,14,16,18,19,20,21,25],easier:19,edg:21,effect:[7,10,13,14,16,18,21,22],effienc:3,either:[3,4,7,10,11,14,16,19,21,22],element:[1,3,11,19,22,25],elementwis:11,elig:16,els:20,emb:19,emo:13,emp:14,empti:[20,21],emv:[7,13],enabl:[3,16,19,21],enable_control:21,encod:16,enforc:[7,13,14],engin:4,enrti:7,enter:16,entir:16,entri:[],equal:[1,11,16,19,21,25],error:[],error_arrai:25,error_r:4,error_sign:25,esp:21,establish:20,estim:4,etc:[4,5,11,16],evalu:[3,11,16,18],evc:[2,3,5],even:[16,18,19],ever:19,everi:[4,5,11,16,19,21],exactli:11,exampl:[11,16,19,21,22],except:[3,7,13,18,19,20],exece:22,execut:[],execution_graph_mech:21,execution_set:21,executiongraph:[2,21],executionlist:[19,21],exist:[11,16],expect:11,explain:5,explan:[11,16],explant:16,explic:10,explicilti:16,explicit:[3,7,10,13,14],explicitli:[1,2,3,4,7,10,13,14,16,20,21,25],expon:[2,21],exponenti:[2,3,21,22],extend:2,extrem:19,factor:[11,19],factori:[11,16,21],fail:20,fall:21,fals:[11,14,16,18,19,20,21],fast:4,faster:22,feedback:[16,21],few:19,field:[3,5,11,16],figur:[11,16,19],first:[2,4,7,10,11,13,14,16,19,20,21,22],fix:10,flat_output:21,float64:22,follow:[1,2,4,7,11,13,14,16,18,19,20,21,22,25],forc:[4,16,19,21],form:[4,11,14,16,19,20],formal:4,format:[3,5,11,16,19,21],forth:19,four:19,framework:[16,19,21],from:[1,2,3,4,7,10,11,13,14,16,18,20,21,22,25],full:[11,13],full_connectivity_matrix:16,fulli:[19,21],function__param:11,function_nam:3,function_param:[1,2,3,4,7,10,11,13,14,16,20],function_run_time_parm:[1,4,25],functioncategori:20,functionnam:3,functionparam:1,functionparrameterst:10,functionpreferenceset:[16,21],functiontyp:[1,2,3,4,5,7,10,11,13,14,16,20,25],functon_param:11,fundament:11,further:[11,19],fuss:4,gain:[11,22],gate:18,gaussian:22,gener:[1,3,4,11,14,16,19,20,25],get:[7,14,18,20],get_adjust:3,get_cost:3,get_duration_cost:3,get_ignoreintensityfunct:3,get_intensity_cost:3,give:4,given:[11,16,19,20],granular:[4,22],graph:[],green:11,hadamard:[1,11],handl:[1,4,10,11,19,20,25],hard_clamp:16,have:[7,10,11,13,16,18,19,20,21,22],help:19,here:[1,3,13,14],hierarch:[16,21],higher:[19,22],histori:3,holm:4,how:[2,3,11,16,19,20],howev:[11,16,19,21],hyphen:[1,3,4,7,10,13,14,20,25],ident:[3,13],identifi:[3,20],identity_matrix:[10,13,16],identitymap:[3,10],identitymatrix:10,ignor:[2,3,16,19,20,21],ignoreintensityfunct:3,impact:[11,21],implement:[1,3,4,5,7,10,11,13,14,16,18,19,20,22,25],implementt:10,includ:[3,7,10,11,13,14,16,18,20,21,22],increment:[7,13,20],index:[1,3,4,7,10,11,13,14,16,18,20,21,22,25,26],indic:[],individu:21,infer:[7,13,14,18],influenc:11,inform:16,inherit:14,init:[7,10,13,14,18],initalize_cycl:21,initi:[],initial_cycl:21,initial_valu:[16,19,21,22],initialize_cycl:[19,21],inlin:[11,16],inner:19,input:[],input_arrai:21,input_st:[2,7,11,13],input_state_nam:2,input_state_param:11,input_state_valu:2,inputst:[],inputstateparam:7,inputvalu:[4,11,16,21,22],insid:19,inspect:21,inspectionopt:21,inspectopt:21,instal:4,instanc:[1,2,3,4,5,7,10,11,13,14,16,18,20,21,22,25],instanti:[1,2,3,4,5,7,10,11,13,14,16,18,20,21,22,25],instantiate_control_mechanism_input_st:2,instantiate_control_signal_channel:5,instantiate_control_signal_project:[2,5],instantiate_output_st:13,instantiate_parameter_st:14,instantiate_projection_from_st:20,instantiate_projections_to_st:20,instantiate_st:[7,13,20],instantiate_state_list:[13,14,20],instead:20,insur:[7,14,19],intact:[16,19,21],intantiate_mechanism_st:20,intens:[2,3],intensity_cost:3,intensitycost:3,intensitycostfunct:3,intercept:[2,3,11],interfac:3,intern:[4,11,16,19,21],interpos:16,interv:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,19,20,21,22,25],intial_valu:19,intiti:4,intreg:4,invalid:20,invok:19,involv:16,irrespect:16,is_pref_set:[1,2,3,4,5,7,10,13,14,16,21,22,25],is_projection_spec:18,isn:[3,10],item:[1,2,3,7,11,13,14,16,19,20,21,22],iter:19,itself:[2,11,16,21],journal:4,just:[2,7,10,11,13,14,20],karg:20,keep:5,kei:[3,7,10,11,13,14,16,18,19,20,21],kept:[1,4,25],keypath:[10,20],keyword:[3,11,16,18,20,21,22],know:5,kvo:20,kwallocationsampl:3,kwaritment:[7,14],kwbogaczet:4,kwcontrolsignaladjustmentcostfunct:3,kwcontrolsignalcost:3,kwcontrolsignalcostfunct:3,kwcontrolsignalcostfunctionnam:3,kwcontrolsignaldefaultnam:3,kwcontrolsignaldurationcostfunct:3,kwcontrolsignalident:3,kwcontrolsignalintensitycostfunct:3,kwcontrolsignallogprofil:3,kwcontrolsignaltotalcostfunct:3,kwddm_analyticsolut:4,kwddm_bia:4,kwexecut:16,kwinputst:[7,10,13,20],kwinputstatevalu:[2,5],kwintegr:4,kwlearn:20,kwlinearcombinationfunct:[7,14],kwlinearcombinationoper:10,kwmappingfunct:10,kwmechan:20,kwmechanisparameterst:14,kwmonitoredst:2,kwmstateproject:20,kwnavarroandfuss:4,kwnavarrosandfuss:4,kwoutputst:20,kwoutputstatevalu:[2,5],kwprojectionparam:18,kwprojectionsend:[3,10,18],kwprojectionsendervalu:[3,10,18],kwreceiv:10,kwstate:20,kwstatefunctioncategori:20,kwstateparam:20,kwstateprojectionaggregationfunct:7,kwstateprojectionaggregationmod:7,kwstatevalu:20,kwtimescal:[1,4],label:[11,21],lambda:[10,11,14],last:[3,13,16,19],last_alloc:3,last_intens:3,later:[1,4,25],latter:[1,4,11,19,25],layer:[16,25],lazi:16,learnin:16,learning_projection_receiv:21,learning_sign:10,learningmechan:[],learningsign:[11,16,20,25],least:[11,16,19,25],left:[16,19,21],length:[1,7,11,13,16,19,20,21,25],level:[4,11,16,19,21],like:[11,19],linear:[2,3,5,11,22],linearcombin:[1,3,7,11,13,14,20],linearmatrix:[10,13],link:[11,16,18,19,21,22],list:[2,3,7,10,11,13,14,16,18,19,20,21,22],local:1,log:[],log_all_entri:3,log_profil:3,logist:[11,16,22],logprofil:3,loop:[16,19,21],lower:22,lowest:[16,19,21],made:[11,16],mai:[5,7,11,14,16],maintain:[1,3,4,7,10,13,14,20,25],make:[4,16,20],make_default_control:2,manag:19,mani:11,manner:21,map:[],mapping_param:11,mappingpreferenceset:10,match:[7,13,16,19,20,25],mathemat:[4,22],matlab:4,matrix:[10,13,16,25],mattrix:16,maximum:[21,22],mean:[1,4,11,22],mech:21,mech_spec:[11,22],mech_tupl:21,mechahn:11,mechainsm:11,mecham:2,mechan:[],mechanim:[16,18,19],mechanism_1:16,mechanism_2:16,mechanism_3:16,mechanism_bas:11,mechanism_specifying_paramet:11,mechanism_st:20,mechanism_typ:11,mechanismlist:[11,16,21],mechanismnam:16,mechanismregistri:[1,4,11,22,25],mechanismsdict:21,mechanismsinputst:1,mechanismsparameterst:20,mechanismtupl:[11,16],mechansim:20,mechansim_1:16,member:16,mention:11,messag:[16,20],met:19,method:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,25],michael:4,mind:19,minimum:22,miss:16,mode:[19,21],model:4,modifi:[1,3,4,11,16,21,25],modul:[3,10,11,16,18,21,26],modulationoper:[10,14],moehli:4,monitored_output:21,monitored_output_label:21,monitored_output_st:[2,11,21],monitored_outputst:21,monitoredoutputst:2,monitoredoutputstatesopt:[2,21],monitoredstatechang:12,monitoring_mechan:21,monitoringmechan:[1,10,12,16,19,21,25],monitoringmechanism_bas:12,more:[5,7,11,13,14,16,19,20,21],most:[3,19],move:2,much:19,multi:[7,13,20],multipl:[2,7,10,11,14,16,19,20,21],multipli:14,must:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],my_linear_transfer_mechan:22,my_logistic_transfer_mechan:22,my_mechan:11,my_process:16,name:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],namedtupl:[11,16],navarro:4,navarroandfuss:4,navarrosandfuss:4,ndarrai:[11,16,19,20,21,22],necessari:[7,13,20],necessarili:[11,16,21],need:[2,4,5,16],nest:[16,19,21],network:16,never:[11,16,18,20,21],nevertheless:19,next:[16,19,21,25],next_level_project:25,nice:[3,5,11,16],node:21,nois:[4,16,22],non:[4,11],non_decision_tim:4,none:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,19,20,21,22,25],notat:14,note:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,25],notimpl:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,22,25],nth:[11,16],num_execut:[11,16,19,21],num_phases_per_tri:21,number:[2,3,4,5,7,10,11,13,14,16,19,20,21,22,25],numer:[11,19,22],numphas:[16,21],object:[1,2,3,4,7,11,13,14,16,18,19,20,21,22],observ:20,obviou:[18,20],occupi:21,occur:21,off:[16,19,21],offer:[11,20],omit:[1,3,4,7,10,13,14,16,20,25],onc:[11,16,19],oni:4,onli:[2,4,7,11,13,16,18,19,20,21,22],oper:[1,7,13,14,18],optim:4,option:[1,2,3,4,11,16,18,19,20,21,22,25],order:[],ordereddict:[7,11,13,20,21],ordereddictionari:20,orderli:21,organ:21,origin:[11,16,19,21],origin_mechan:21,originmechan:[16,21],other:[2,3,4,10,11,14,16,19,21,25],otherwis:[3,7,10,11,13,16,18,19,20],ou_upd:4,ouput:11,ouputst:5,out:[11,20],outcom:[1,4,11],outermost:[11,16,19,21],output:[],output_st:[4,11,13],output_state_nam:21,output_state_param:11,output_value_arrai:21,outputindex:[3,5],outputst:[],outputstatenam:11,outputstatevalu:11,outputvalu:[3,5,11,21,22],outsid:21,outstat:[2,21],over:[11,16],overrid:[2,5,10,11,16,18,21],overridden:[1,2,4,11,21,25],overriden:2,own:[7,10,11,13,16],owner:[3,7,11,13,14,16,20],page:26,pair:16,param:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],param_default:[1,4,25],param_modulation_oper:10,param_nam:20,paramclassdefault:[1,2,3,4,5,7,10,13,14,18,20,25],parameat:11,paramet:[],parameter:21,parameter_modulation_oper:14,parameter_spec:2,parameter_st:14,parameter_state_param:[10,11,14,16],parameterst:[],parameterstateparam:14,paraminstancedefault:[1,4,7,10,13,14,18,25],parammodulationoper:14,paramnam:[1,3,4,7,10,13,14,18,20,25],params_dict:20,paramscurr:[1,4,7,10,13,14,18,20,25],paramvalueproject:[7,11,14,20],parm:[18,22],pars:[14,19],parse_projection_ref:20,part:[3,7,10,11,13,14,16,18,20],particl:4,particular:[11,16],pass:[3,4,5,7,10,11,14,20,21,22],passag:4,path:21,pathwai:[],pattern:16,per:[4,19],perform:[4,11],permiss:21,permit:21,phase:[],phasespec:[11,19],physic:4,pickl:[3,5,11,16],place:[7,13,16,20],plai:[11,21],plain:[3,5,11,16],plot:2,point:[7,13,21],popul:18,posit:21,possibl:[10,19],preced:[7,10,11,16],pref:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,20,21,22,25],prefer:[],preferenceentri:[10,20],preferencelevel:[1,3,4,10,14,20,25],preferenceset:[1,3,4,10,11,16,18,20,21,22,25],prefix:20,present:[16,19],preserv:16,previou:[11,22],primari:[2,11,16,19,21],primarili:[13,16],primary_output:21,primary_output_label:21,primary_output_st:[2,21],print:[2,21],prior:19,probability_lower_bound:4,probability_upper_bound:4,process:[],process_bas:16,process_input:16,process_spec:16,processingmechan:[16,17,21,25],processingmechanism_bas:17,processinputst:16,processlist:16,processnam:16,processregistri:16,processtupl:16,produc:[7,14],product:[11,13,14],project:[],projection_a:16,projection_bas:18,projection_param:[11,18,20],projection_spec:20,projection_typ:[14,18,20],projectionpreferenceset:[],projectionregistri:[3,10,18],projectionsend:18,projectiontyp:14,protocol:2,provid:[1,3,4,7,11,13,16,18,19,20,21,22,25],prune:21,psycholog:4,psyneulink:[],psyneulnk:11,purpl:11,purpos:[11,19],python:11,question:1,quotient:1,rais:[3,7,13,18,20],random:[1,2,3,4,5,7,10,11,12,13,14,16,17,18,19,20,21,22,25],random_connectivity_matrix:16,rang:22,rate:[4,11,22],rather:[13,16,18,20],reaction:4,read:21,real:[19,21],reassign:20,receiv:[3,7,10,11,14,16,18,20,21,25],receivesfromproject:[11,20],recent:3,recurr:[16,21],recurrent_init_arrai:21,recurrent_mechan:21,recurrentinitmechan:21,red:11,redund:10,ref:[2,3,7,10,11,14,20],refer:[1,14,16,18,20],referenc:21,reference_valu:[7,13,14],referenceto:2,regist:[1,3,4,7,10,11,13,14,20,25],register_categori:[1,3,4,10,20,25],registri:[11,16,18,21,22],reinforc:16,relabel:14,relev:[11,13,16,19,22],remain:[16,21],remov:21,report:3,repres:[7,11,13,14,20,21],represent:[3,5,11,16],request:[2,20],request_set:[2,25],requir:[3,4,7,10,11,13,14,16,19,21],requiredparamclassdefaulttyp:20,reset:[16,19,21],reset_clock:[16,19,21],resolv:22,respect:[11,21],respons:[4,11,19],restrict:[1,4],result:[2,7,11,13,14,16,19,20,21,22],result_mean:22,result_vari:22,review:4,round:[16,19,21],rouund:19,row:25,rt_correct_mean:4,rt_correct_vari:4,rt_mean:4,rtype:[1,3,4,22],run:[],runtim:[],runtime_param:[2,4,11,16,22],same:[1,10,11,13,14,16,18,19,20,21,25],sampl:[1,3,11,21],sample_input:19,scalar:[2,3,19,22],scale:[16,19,22],schemat:11,scope:19,search:[3,10,26],second:[14,16,20,22],see:[1,3,4,7,10,11,13,14,16,18,19,20,21,22,25],select:11,self:[1,2,3,4,5,7,10,11,12,13,14,16,20,21,22,25],send:[13,16,18,21],sender:[2,3,5,7,10,11,13,14,16,18,19,20,21,25],senderdefault:10,sendstoproject:[18,20],separ:[1,4,25],sequenc:[11,16,19,21],sequeunc:19,seri:16,serv:[11,18],set:[1,3,4,7,10,11,12,13,14,16,18,19,20,21,22,25],set_adjustment_cost:3,set_allocation_sampl:3,set_duration_cost:3,set_intensity_cost:3,set_log:3,set_log_profil:3,set_valu:20,sever:[1,3,4,10,11,21,25],share:11,should:[3,5,10,11,13,14,16,18,19,20,21,22],show:[2,11,19,21],shown:11,shvartsman:4,sigmoid:2,signal:[],similar:11,similarli:11,simpl:22,simpler:19,simplest:[11,19],simpli:[5,11,13,19,22],simul:4,sinc:[18,20,21],singl:[2,3,4,7,10,11,13,14,16,18,19,20,21,22],singleton:21,size:19,slope:[2,3,11],soft_clamp:16,softmax:2,sole:[7,13,20],solut:4,some:[11,16,19,21],some_param:16,someth:5,sophist:5,sort:21,sourc:[3,5,10,16,19],spec:[16,18,20,21],special:[4,7,11,13],specif:[2,3,7,10,11,13,14,16,18,19,20,21,22],specifi:[],specifii:11,squar:[1,25],squqr:1,stand:[16,19],standard:[1,4,11,18,20,22,25],start:[16,21,22],starting_point:4,state:[],state_bas:[7,13,20],state_list:20,state_nam:20,state_param:20,state_param_identifi:20,state_project:[7,14,20],state_spec:[2,20],state_typ:20,statepreferenceset:20,stateregistri:[7,13,14,20],statist:[1,25],std:[1,25],step:[4,11,16,19,21],stepwis:4,still:[3,10,16,20],stochast:4,stop:4,store:[11,16,19,21],str:[1,2,3,4,5,7,10,11,13,14,16,18,20,21,22,25],string:[3,5,11,16,18,20,21,22],sub:11,subclass:[],sublist:19,submit:11,subsequ:[13,16],subset:21,subtract:1,subtyp:[1,25],suffix:[1,3,4,7,10,11,13,14,16,20,25],sum:[1,7,10,11,13,25],summari:[1,25],support:[7,11,13,16,19,21],suppress:20,synonym:[11,16],system:[],system_bas:21,systemcontrol:2,systemdefaultccontrol:2,systemdefaultreceiv:10,systemdefaultsend:10,systemregistri:21,take:[3,11,16,18,20,21],take_over_as_default_control:2,target:[],target_input:19,target_set:[2,25],task:[3,4],tbi:[2,3,4,18,21],tc_predic:1,templat:[11,16],tempor:[3,4,22],term:[4,19],termin:[1,2,4,11,16,19,21],terminal_mechan:21,terminalmechan:[16,21],terminate_funct:[1,4],test:[1,11,12,16,20],than:[1,4,7,11,13,14,16,18,19,20,25],thei:[1,4,7,11,14,16,21,22,25],them:[2,7,10,13,14,16,20],themselv:[11,16,21],theoret:21,therefor:16,thi:[1,2,3,4,5,7,10,11,13,14,16,18,19,20,21,22,25],third:16,those:[1,3,4,5,7,10,11,13,14,16,19,20,21,25],though:16,three:[3,11,16,18,20,21],threshold:[4,16],through:[16,21],throughout:16,thu:11,time:[],time_scal:[1,2,3,4,10,11,14,16,19,20,21,22,25],time_step:[4,11,16,19,21,22],time_step_s:4,timescal:[1,2,4,11,14,16,19,20,21,22,25],togeth:21,tool:21,topolog:21,toposort:21,total:[3,4],total_alloc:4,total_cost:4,total_cost_funct:3,totalcost:3,totalcostfunct:3,track:5,train:[11,16],trajectori:4,transfer:[],transfer_default_bia:22,transferouput:22,transform:[2,10,11,16,21,22],translat:3,transmit:[11,16],treatment:19,trial:[1,2,4,11,14,16,19,20,21,22,25],tupl:[2,3,5,11,14,16,20,21,22],tuples_list:[11,16],turn:[11,20],two:[1,4,7,11,13,14,16,18,19,20,21,25],type:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],typecheck:[1,16,19,21],typic:[11,19],unchang:[5,10],under:[10,11,16,18,20,21,22],understand:19,uniqu:3,unit:22,unless:[1,4,14,25],until:[1,4,11,16,25],updat:[2,3,4,7,10,11,13,14,16,19,20],update_control_sign:3,update_monitored_state_changed_attribut:12,update_st:[7,14,20],upon:19,upper:22,user:20,usual:[11,19],util:[],utilityfunct:22,valid:[7,10,11,13,14,18,19,20,25],validate_monitoredstates_spec:2,valu:[],variabilti:4,variabl:[1,2,3,4,7,10,11,12,13,14,16,17,18,19,20,22,25],variable_default:2,variableclassdefault:[1,4,20,25],variableindex:[3,5],variableinstancedefault:[11,16],variablevalu:[3,5],varianc:[1,11,22,25],variou:11,vector:[1,3,10,19],version:4,wai:[1,3,4,7,10,11,13,14,16,18,20,21,25],warn:20,weight:[],weightederror:25,weightederror_preferenceset:25,well:[1,4,11,25],what:[1,5],when:[2,7,11,13,16,19,21,22],whenev:16,where:[11,20,21],whether:[1,2,11,12,16,18,19,20,21,22],which:[1,2,3,4,7,10,11,13,14,16,18,19,20,21,22,25],whose:21,why:3,width:25,wiener:4,wise:[1,4,22],within:[7,10,13,16,19,20,21],without:20,would:[11,19],xxx:[14,16,20],zero:[16,21,22]},titles:["Adaptive Integrator","Comparator","Control Mechanisms","Control Signal","DDM","Default Control Mechanism","&lt;no title&gt;","Input State","Learning","Log","Mapping","Mechanisms","Monitoring Mechanisms","Output State","Parameter State","Preferences","Process","Processing Mechanisms","Projections","Run","States","System","Transfer","Utilities","Utility Functions","Weighted Error","Welcome to PsyNeuLink&#8217;s documentation!"],titleterms:{"default":5,"function":[7,11,24],adapt:0,compar:1,control:[2,3,5,21],controlmechan:2,creat:[2,7,11,13,14,16,18,21,22],ddm:4,document:26,entri:9,error:25,execut:[11,16,18,21,22],graph:21,indic:26,initi:[19,21],input:[7,16,19,21],inputst:[7,11],integr:0,learn:[8,16,21],log:9,map:10,mechan:[2,5,11,12,16,17,21,22],monitor:[2,12],order:21,output:[13,16],outputst:[2,11,13],overview:[2,7,11,14,16,18,19,21,22],paramet:[11,14],parameterst:[11,14],pathwai:16,phase:21,prefer:15,process:[11,16,17],project:[16,18],psyneulink:26,role:11,run:19,runtim:11,signal:3,specifi:11,state:[7,11,13,14,20],structur:[16,21],subclass:18,system:[11,21],tabl:26,target:19,time:19,transfer:22,util:[23,24],valu:19,weight:25,welcom:26}})