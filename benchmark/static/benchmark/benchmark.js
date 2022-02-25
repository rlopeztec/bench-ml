function changeOption(idValue){
  switch (idValue) {
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
            + '<td width="34%" valign="top">'
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
            + '</td>'
          + '</tr>'
        + '</table>';
      break;
    case 'Conv1D':
      document.getElementById("selectOption").innerHTML = ''
        + '<table width="100%" style="margin-left:14px;">'
          + '<tr>'
            + '<td width="55%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="1" checked>'
                + ' option 1<br>'
                + 'Conv1D(filters=[f], 2, activation=[afn])<br>'
                + 'MaxPooling1D()<br>'
                + 'Flatten()<br>'
                + 'Dense(filters=[f]/2 activation=[afn])<br>'
                + 'Dense(num_classes, activation=[aout])<br>'
                + 'model.compile(<br>'
                       + '&nbsp&nbsp&nbsp [optimizer](lr=[lr]),<br>'
                       + '&nbsp&nbsp&nbsp loss=[loss],<br>'
                       + '&nbsp&nbsp&nbsp metrics=[accuracy])'
            + '</td>'
            + '<td width="45%" valign="top">'
              + '<input type="radio" id="model_option" name="model_option" value="2">'
              + ' option 2<br>'
            + '</td>'
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
  if (['FC','Conv1D'].includes(idValue)) {
      document.getElementById("selectHTML").innerHTML = ''
        + '<div class="col-md-6 mb-3">'
          + '<label for="filters">Number of filters: </label>'
          + '<input type="text" id="filters" name="filters" value="128" required>'
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="activation">Activation function: </label>'
          + '<select id="activation" name="activation" required>'
            + '<option value="relu" selected>relu</option>'
            + '<option value="elu">elu</option>'
            + '<option value="tanh">tanh</option>'
            + '<option value="linear">linear</option>'
            + '<option value="sigmoid">sigmoid</option>'
            + '<option value="swish">swish</option>'
          + '</select>'
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="activation_output">Activation output: </label>'
          + '<select id="activation_output" name="activation_output" required>'
            + '<option value="softmax" selected>softmax</option>'
            + '<option value="sigmoid">sigmoid</option>'
          + '</select>'
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="optimizer">Optimizer: </label>'
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
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="learning_rate">Learning rate: </label>'
          + '<input type="text" id="learning_rate" name="learning_rate" value="0.0003" required>'
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="loss">Loss: </label>'
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
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="dropout">Dropout: </label>'
          + '<input type="text" id="dropout" name="dropout" value="0.30" required>'
          + ' if 0 is not used'
        + '</div>'
        + '<div class="col-md-6 mb-3">'
          + '<label for="earlystop">Early stop: </label>'
          + '<input type="text" id="earlystop" name="earlystop" value="25" required>'
          + ' if 0 is not used'
        + '</div>';
        // TODO? add MinMaxScaler along with StandardScaler
  } else {
    switch (idValue) {
      case 'DT':
        document.getElementById("selectHTML").innerHTML = '';
        break;
      case 'RF':
        document.getElementById("selectHTML").innerHTML = ''
          + '<div class="col-md-6 mb-3">'
            + '<label for="estimators">Estimators to run: </label>'
            + '<input type="text" id="estimators" name="estimators" value="100" required>'
            + '<div class="invalid-feedback">'
              + 'Enter number of estimators to run i.e. 100'
            + '</div>'
          + '</div>';
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
