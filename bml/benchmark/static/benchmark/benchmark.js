// when training and running a model validate all fields
function validateRunModel(){
	if (!validateInteger('filters')) { return false;}
	// if Conv1D
	if (document.getElementById('model_ran').value == 'Conv1D'){
		if (!validateInteger('kernels')) { return false;}
		if (!validateInteger('poolsize')) { return false;}
	}
	if (!validateNumber('learning_rate')) { return false;}
	if (!validateNumber('dropout')) { return false;}
	if (!validateInteger('earlystop')) { return false;}
	if (!validateInteger('epochs')) { return false;}
}

// validate is an integer
function validateInteger(field){
	//return event.charCode >= 48 && event.charCode <= 57
	fvalue = document.getElementById(field).value;
	if (isNaN(fvalue)) {
		alert(field + " not an integer");
	        document.getElementById(field).focus();
		return false;
	} else {
		fvalue = parseFloat(fvalue);
		if (fvalue < 0 || fvalue > 9999) {
		    alert(field + " outside range 0 and 9999: " + fvalue);
	            document.getElementById(field).focus();
		    return false;
		} else {
			return true;
		}
	}
}
// validate is a real number between 0 and 0.99999
function validateNumber(field){
	fvalue = document.getElementById(field).value;
	if (isNaN(fvalue)) {
		alert(field + " not a number between 0 and 0.99999");
	        document.getElementById(field).focus();
		return false;
	} else {
		if (fvalue < 0 || fvalue > 0.99999) {
		    alert(field + " outside range 0 and 0.99999");
	            document.getElementById(field).focus();
		    return false;
		} else {
			return true;
		}
	}
}

function changeOption(idValue){
  switch (idValue) {
    case 'BNN':
    case 'IBNN':
    case 'FC':
      document.getElementById("selectOption").innerHTML = ''
        + '<table width="100%" style="margin-left:14px;">'
          + '<tr>'
            + '<td width="22%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="1">'
                + ' option 1<br>'
                + 'Dense(filters=[f] activation=[afn])<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn])<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(#classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<td width="22%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="2" checked>'
                + ' option 2<br>'
                + 'Dense(filters=[f] activation=[afn])<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/2)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/4)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/6)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(num_classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<td width="22%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="3">'
                + ' option 3<br>'
                + 'Dense(filters=[f] activation=[afn])<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/2)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/4)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/6)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/8)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(filters=[f] activation=[afn]/10)<br>'
                + 'Dropout([dropout])<br>'
                + 'Dense(num_classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<!--td width="34%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="4">'
                + ' option 4<br>'
                + '<textarea id="model_text" name="model_text" rows="15" cols="45">'
                + 'Dense(filters=[f] activation=[act fn])\n'
                + 'Dropout([dropout])\n'
                + 'Dense(filters=[f] activation=[act fn]/2)\n'
                + 'Dropout([dropout])\n'
                + 'Dense(filters=[f] activation=[act fn]/4)\n'
                + 'Dropout([dropout])\n'
                + 'Dense(num_classes, activation=[aoutput])\n'
                + 'model.compile(\n'
                       + '\t[optimizer](lr=[lr]),\n'
                       + '\tloss=[loss],\n'
                       + '\tmetrics=[accuracy])'
                + '</textarea>'
            + '</td--!>'
          + '</tr>'
        + '</table>';
      break;
    case 'Conv1D':
      document.getElementById("selectOption").innerHTML = ''
        + '<table width="100%" style="margin-left:14px;">'
          + '<tr>'
            + '<td width="22%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="1" checked>'
                + ' option 1<br>'
                + 'Conv1D([filters], [kernels], activation=[afn])<br>'
                + 'MaxPooling1D([poolsize])<br>'
                + 'Flatten()<br>'
                + 'Dense(filters=[f]/2 activation=[afn])<br>'
                + 'Dense(num_classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<td width="22%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="2">'
              + ' option 2<br>'
                + 'Conv1D([filters], [kernels], activation=[afn])<br>'
                + 'MaxPooling1D([poolsize])<br>'
                + 'Conv1D(filters=[f]/2, [kernels], activation=[afn])<br>'
                + 'MaxPooling1D([poolsize])<br>'
                + 'Flatten()<br>'
                + 'Dense(filters=[f]/4 activation=[afn])<br>'
                + 'Dense(num_classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<td width="22%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="3">'
              + ' option 3<br>'
                + 'Conv1D([filters], [kernels], activation=[afn])<br>'
                + 'MaxPooling1D([poolsize])<br>'
                + 'Conv1D(filters=[f]/2, [kernels], activation=[afn])<br>'
                + 'MaxPooling1D([poolsize])<br>'
                + 'Conv1D(filters=[f]/4, [kernels], activation=[afn])<br>'
                + 'MaxPooling1D([poolsize])<br>'
                + 'Flatten()<br>'
                + 'Dense(filters=[f]/8 activation=[afn])<br>'
                + 'Dense(num_classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<!--td width="36%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="4">'
              + ' option 4<br>'
                + '<textarea id="model_text" name="model_text" rows="15" cols="45">'
                + 'Conv1D([filters], [kernels], activation=[afn])\n'
                + 'MaxPooling1D([poolsize])\n'
                + 'Flatten()\n'
                + 'Dense(filters=[f]/2 activation=[afn])\n'
                + 'Dense(num_classes, activation=[aout])\n'
                + 'model.compile(\n'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),\n'
                       + '&nbsp&nbsp&nbsp loss=[loss],\n'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
                + '</textarea>'
            + '</td--!>'
          + '</tr>'
        + '</table>';
      break;
    case 'DT':
      document.getElementById("selectOption").innerHTML = '';
      break;
    case 'RF':
      document.getElementById("selectOption").innerHTML = '';
      break;
    case 'SVM':
      document.getElementById("selectOption").innerHTML = '';
      break;
    case 'GaussianNB':
      document.getElementById("selectOption").innerHTML = '';
      break;
  }
}

function changeSelect(idValue){
      if (document.getElementById("hepochs").value) {
        hepochs=document.getElementById("hepochs").value;
      } else {
        hepochs="20"
      }
      if (document.getElementById("hfilters").value == null) {
        hfilters="32"
      } else {
        hfilters=document.getElementById("hfilters").value;
      }
      if (document.getElementById("hkernels")) {
        hkernels=document.getElementById("hkernels").value;
      } else {
        hkernels="2"
      }
      if (document.getElementById("hpoolsize")) {
        hpoolsize=document.getElementById("hpoolsize").value;
      } else {
        hpoolsize="1"
      }
      if (document.getElementById("hlearning_rate")) {
        hlearning_rate=document.getElementById("hlearning_rate").value;
      } else {
        hlearning_rate="0.003"
      }
      if (document.getElementById("hdropout")) {
        hdropout=document.getElementById("hdropout").value;
      } else {
        hdropout="0.30"
      }
      if (document.getElementById("hearlystop")) {
        hearlystop=document.getElementById("hearlystop").value;
      } else {
        hearlystop="25"
      }
  if (['FC'].includes(idValue) || ['BNN'].includes(idValue) || ['IBNN'].includes(idValue)) {
      document.getElementById("selectHTML").innerHTML = ''
        + '<table width="100%">'
        + '<tr>'
        + '<td width="23%" style="padding: 0px 0px 0px 15px;">'
          + '<label for="filters">Number of filters: </label>'
        + '</td>'
        + '<td width="23%">'
          + '<input type="text" id="filters" name="filters" value="' + hfilters +'" onkeypress="return event.charCode >= 48 && event.charCode <= 57">'
        + '</td>'
        + '<td width="04%">&nbsp;'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Number of filters from 1 to 9999, it should be less than number of features/dimensions</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td width="23%">'
          + '<label for="activation">Activation function: </label>'
        + '</td>'
        + '<td width="23%">'
          + '<select id="activation" name="activation" required>'
            + '<option value="relu" selected>relu</option>'
            + '<option value="elu">elu</option>'
            + '<option value="tanh">tanh</option>'
            + '<option value="linear">linear</option>'
            + '<option value="sigmoid">sigmoid</option>'
            + '<option value="swish">swish</option>'
          + '</select>'
        + '</td>'
        + '<td width="04%"> &nbsp; </td>'
        + '</tr>'
        + '<tr>'
        + '<td width="23%" style="padding: 0px 0px 0px 15px;">'
          + '<label for="activation_output">Activation output: </label>'
        + '</td>'
        + '<td width="23%">'
          + '<select id="activation_output" name="activation_output" required>'
            + '<option value="softmax" selected>softmax</option>'
            + '<option value="sigmoid">sigmoid</option>'
          + '</select>'
        + '</td>'
        + '<td width="04%"></td>'
        + '<td width="23%">'
          + '<label for="optimizer">Optimizer: </label>'
        + '</td>'
        + '<td width="23%">'
          + '<select id="optimizer" name="optimizer" required>'
            + '<option value="Adam" selected>Adam</option>'
            + '<option value="Adadelta">Adadelta</option>'
            + '<option value="Adagrad">Adagrad</option>'
            + '<option value="Adamax">Adamax</option>'
            + '<option value="Ftrl">Ftrl</option>'
            + '<option value="Nadam">Nadam</option>'
            + '<option value="RMSprop">RMSprop</option>'
            + '<option value="SGD">SGD</option>'
          + '</select>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="learning_rate">Learning rate: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="learning_rate" name="learning_rate" value="' + hlearning_rate + '" onchange="validateNumber(\'learning_rate\')" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Value should be from 0.00001 to 0.99999</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td>'
          + '<label for="loss">Loss: </label>'
        + '</td>'
        + '<td>'
          + '<select id="loss" name="loss" required>'
            + '<option value="categorical_crossentropy" selected>categorical crossentropy</option>'
            + '<option value="binary_crossentropy">binary_crossentropy</option>'
            + '<option value="MSE">MSE</option>'
            + '<option value="MSLE">MSLE</option>'
            + '<option value="kullback_leibler_divergence">kullback_leibler_divergence</option>'
            + '<option value="mean_absolute_error">mean_absolute_error</option>'
            + '<option value="mean_squared_error">mean_squared_error</option>'
            + '<option value="mean_squared_logarithmic_error">mean_squared_logarithmic_error</option>'
            + '<option value="poisson">poisson</option>'
            + '<option value="squared_hinge">squared_hinge</option>'
          + '</select>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="dropout">Dropout: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="dropout" name="dropout" value="' + hdropout + '" onchange="validateNumber(\'dropout\')" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Value from 0 to 0.99, if 0 dropout is not used</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td>'
          + '<label for="earlystop">Early stop: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="earlystop" name="earlystop" value="' + hearlystop + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Only integers, if 0 early stop is not used</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="epochs">Epochs: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="epochs" name="epochs" value="' + hepochs + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Integers equal or greater than 1</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '</tr>'
        + '</table>';
        // TODO? add MinMaxScaler along with StandardScaler
  } else if (['Conv1D'].includes(idValue)) {
      document.getElementById("selectHTML").innerHTML = ''
        + '<table width="100%">'
        + '<tr>'
        + '<td width="23%" style="padding: 0px 0px 0px 15px;">'
          + '<label for="filters">Number of filters: </label>'
        + '</td>'
        + '<td width="23%">'
          + '<input type="text" id="filters" name="filters" required value="' + hfilters + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57">'
        + '</td>'
        + '<td width="04%">'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Number of filters from 1 to 9999, it should be less than number of features/dimensions</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td width="23%">'
          + '<label for="activation">Activation function: </label>'
        + '</td>'
        + '<td width="23%">'
          + '<select id="activation" name="activation" required>'
            + '<option value="relu" selected>relu</option>'
            + '<option value="elu">elu</option>'
            + '<option value="tanh">tanh</option>'
            + '<option value="linear">linear</option>'
            + '<option value="sigmoid">sigmoid</option>'
            + '<option value="swish">swish</option>'
          + '</select>'
        + '</td>'
        + '<td width="04%"> &nbsp; </td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="kernels">Kernels: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="kernels" name="kernels" required value="' + hkernels + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57">'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Kernels from 1 to 9999, it should be less than number of features/dimensions</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td>'
          + '<label for="poolsize">Max Pooling Size: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="poolsize" name="poolsize" required value="' + hpoolsize + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57">'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>From 1 to N. Reduce dimensions of feature maps to avoid over-fitting and reduce parameters to perform faster</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="activation_output">Activation output: </label>'
        + '</td>'
        + '<td>'
          + '<select id="activation_output" name="activation_output" required>'
            + '<option value="softmax" selected>softmax</option>'
            + '<option value="sigmoid">sigmoid</option>'
          + '</select>'
        + '</td>'
        + '<td>'
        + '</td>'
        + '<td>'
          + '<label for="optimizer">Optimizer: </label>'
        + '</td>'
        + '<td>'
          + '<select id="optimizer" name="optimizer" required>'
            + '<option value="Adam" selected>Adam</option>'
            + '<option value="Adadelta">Adadelta</option>'
            + '<option value="Adagrad">Adagrad</option>'
            + '<option value="Adamax">Adamax</option>'
            + '<option value="Ftrl">Ftrl</option>'
            + '<option value="Nadam">Nadam</option>'
            + '<option value="RMSprop">RMSprop</option>'
            + '<option value="SGD">SGD</option>'
          + '</select>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="learning_rate">Learning rate: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="learning_rate" name="learning_rate" value="' + hlearning_rate + '" onchange="validateNumber(\'learning_rate\')" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Value should be from 0.00001 to 0.99999</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td>'
          + '<label for="loss">Loss: </label>'
        + '</td>'
        + '<td>'
          + '<select id="loss" name="loss" required>'
            + '<option value="categorical_crossentropy" selected>categorical crossentropy</option>'
            + '<option value="binary_crossentropy">binary_crossentropy</option>'
            + '<option value="MSE">MSE</option>'
            + '<option value="MSLE">MSLE</option>'
            + '<option value="kullback_leibler_divergence">kullback_leibler_divergence</option>'
            + '<option value="mean_absolute_error">mean_absolute_error</option>'
            + '<option value="mean_squared_error">mean_squared_error</option>'
            + '<option value="mean_squared_logarithmic_error">mean_squared_logarithmic_error</option>'
            + '<option value="poisson">poisson</option>'
            + '<option value="squared_hinge">squared_hinge</option>'
          + '</select>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="dropout">Dropout: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="dropout" name="dropout" value="' + hdropout + '" onchange="validateNumber(\'dropout\')" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Value from 0 to 0.99, if 0 dropout is not used</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '<td>'
          + '<label for="earlystop">Early stop: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="earlystop" name="earlystop" value="' + hearlystop + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Only integers, if 0 early stop is not used</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '</tr>'
        + '<tr>'
        + '<td style="padding: 0px 0px 0px 15px;">'
          + '<label for="epochs">Epochs: </label>'
        + '</td>'
        + '<td>'
          + '<input type="text" id="epochs" name="epochs" value="' + hepochs + '" onkeypress="return event.charCode >= 48 && event.charCode <= 57" required>'
        + '</td>'
        + '<td>'
        + '<div class="col-md-6 mb-3">'
	  + '<div class="help-tip">'
          + '<p>Integers equal or greater than 1</p>'
          + '</div>'
        + '</div>'
        + '</td>'
        + '</tr>'
        + '</table>';
        // TODO? add MinMaxScaler along with StandardScaler
  } else {
    switch (idValue) {
      case 'DT':
        document.getElementById("selectHTML").innerHTML = '';
        break;
      case 'RF':
        document.getElementById("selectHTML").innerHTML = ''
          + '<table width="100%">'
          + '<tr>'
          + '<td width="23%" style="padding: 0px 0px 0px 15px;">'
            + '<label for="estimators">Estimators to run: </label>'
          + '</td>'
          + '<td width="23%">'
            + '<input type="text" id="estimators" name="estimators" value="100" required>'
          + '</td>'
          + '<td width="54%" colspan="4">'
          + '</td>'
          + '</tr>'
          + '</table>';
        break;
      case 'SVM':
        document.getElementById("selectHTML").innerHTML = '';
        break;
      case 'GaussianNB':
        document.getElementById("selectHTML").innerHTML = '';
        break;
    }
  }
  changeOption(idValue);
}
