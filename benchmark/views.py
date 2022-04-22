# views.py is the view of django
# is quite big <500 lines, i should refactor

import os, glob

# django libraries
from django.shortcuts import render
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views import generic
from django.utils import timezone
from django.core.files.storage import FileSystemStorage
from django.template.defaulttags import register
from django.contrib.auth import logout

# database tables
from .models import Choice, Question, File_raw, Model_Run, Model_Run_Steps, Model_Run_Features
from .models import Classification_Report, Confusion_Matrix, Web_User

# business layer
from .ScoreFeatures import ScoreFeatures
from .SplitTrainTest import SplitTrainTest
from .pca_filter_features import readGenesScoringFile, readHeadersFile, readInputFile
from .ModelConv1D import ModelConv1D
from .ModelFullyConnected import ModelFullyConnected
from .ModelBNN import ModelBNN
from .GaussianNaiveBayes import GaussianNaiveBayes
from .DecisionTree import DecisionTree
from .RandomForest import RandomForest
from .SupportVectorMachine import SupportVectorMachine
from .Pca3D import buildPcaImage

# Create your views here.
class IndexView(generic.ListView):
    template_name = 'benchmark/index.html'
    context_object_name = 'latest_question_list'

    def get_queryset(self):
        return Question.objects.filter(
            pub_date__lte=timezone.now()
        ).order_by('-pub_date')[:5]

class ListView(generic.ListView):
    template_name = 'benchmark/list.html'
    context_object_name = 'latest_question_list'
    active='Compare'

    def get_queryset(self):
        return Question.objects.filter(
            pub_date__lte=timezone.now()
        ).order_by('-pub_date')[:5]

class DetailView(generic.DetailView):
    model = Question
    template_name = 'benchmark/detail.html'
    def get_queryset(self):
        """
        Excludes any questions that aren't published yet.
        """
        return Question.objects.filter(pub_date__lte=timezone.now())

class ResultsView(generic.DetailView):
    model = Question
    template_name = 'benchmark/results.html'

# build pca image file
def buildPca(fileRawId, filename, targetClass):
    nameOnly = filename[filename.rfind('/')+1:len(filename)]
    buildPcaImage(filename, filename + '.genes', 'benchmark/static/benchmark/images/' + nameOnly + '.pca.png', targetClass)

# upload file to be scored
# https://simpleisbetterthancomplex.com/tutorial/2016/08/01/how-to-upload-files-with-django.html
def score(request):
    if 'userid' not in request.session or request.session['userid'] == None:
        return render(request, 'benchmark/signin.html', {
            'message':'Please enter username and password',
            'active':"Signin",
        })
    
    if request.method == 'POST' and request.FILES['myfile']:

        # insert row into database table file_raw
        myfile = request.FILES['myfile']
        r = File_raw(web_user_id=request.session['userid'], file_name=myfile.name, file_desc=request.POST['fileDesc'], number_features=request.POST['numberFeatures'], methods=request.POST['methods'], percent_train=request.POST['trainingPercent'], dist_class=request.POST['distClass'], target_class=request.POST['targetClass'], file_type=request.POST['fileType'], pub_date=timezone.now())
        r.save()
        r.file_name = str(r.id) + '_' + r.file_name
        r.save()

        # upload file
        fs = FileSystemStorage()
        filename = fs.save('benchmark/files/'+r.file_name, myfile)
        uploaded_file_url = fs.url(filename)

        # score file with multiple number of features
        featureFiles = ScoreFeatures.scoreTrain( None, filename, r.file_type, r.number_features, r.methods, r.target_class)

        # split data into train and test each features file
        methodsList = r.methods.split(',')
        featuresList = r.number_features.split(',')

        for method in methodsList:
            for numFeat in featuresList:

                genesScoredFile = filename+'.'+method+'.'+numFeat+'.scores'
                featFile = filename + '.' + method + '.' + numFeat

                #if numFeat != 'all':
                # get top features from scored features
                topGenes = readGenesScoringFile(genesScoredFile, numFeat)
                # create genes files in single line in tsv format to be used in pca
                headersGenes = readHeadersFile(filename, topGenes, featFile+'.genes', r.file_type)
                # creates features file with target class and features selected
                readInputFile(filename, r.file_type, featFile, headersGenes, r.target_class)
                #raise Exception('stopping here')

                # split train/test data
                targetCol, retGenes, retClasses = SplitTrainTest.readGenesScoringFile(None, filename, r.file_type, request.POST['targetClass'])
                linesTraining, linesTest = SplitTrainTest.writeOutputFiles(None, int(r.percent_train), targetCol, retGenes, retClasses, featFile+'.train', featFile+'.test')

                # create pca image file
                buildPca(r.id, featFile, r.target_class)

        return render(request, 'benchmark/score.html', {
            'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']).order_by('-pub_date'),
            'uploaded_file_url': uploaded_file_url,
            'active':"Load",
        })
    return render(request, 'benchmark/score.html', {
        'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']).order_by('-pub_date'),
        'active':"Load",
    })

# run specific model
def specificModel(fileRaw, method, nFeats, modelRan, idModelRunFeatures, parametersList, epochs, targetClass, idFileRaw):

    fileName = 'benchmark/files/'+ fileRaw.file_name + '.' + method + '.' + nFeats
    if modelRan == 'Conv1D':
        classifier = ModelConv1D()
        accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
    else:
        if modelRan == 'FC':
            classifier = ModelFullyConnected()
            accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
        else:
            if modelRan == 'RF':
                classifier = RandomForest()
                accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
            else:
                if modelRan == 'SVM':
                    classifier = SupportVectorMachine()
                    accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
                else:
                    if modelRan == 'GaussianNB':
                        classifier = GaussianNaiveBayes()
                        accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
                    else:
                        if modelRan == 'DT':
                            classifier = DecisionTree()
                            accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
                        else:
                            if modelRan == 'BNN':
                                classifier = ModelBNN()
                                accuracyScore = classifier.runModel(idModelRunFeatures, fileName+'.train', fileName+'.test', parametersList, epochs, targetClass, True, fileRaw.id)
    return accuracyScore

# run ML model
def run(request):
    if 'userid' not in request.session or request.session['userid'] == None:
        return render(request, 'benchmark/signin.html', {
            'message':'Please enter username and password',
            'active':"Signin",
        })
    
    if request.method == 'POST':
        print('runModel', request.POST['idSelected'])
        request.session['idSelected'] = int(request.POST['idSelected'])
        fileRaw = get_object_or_404(File_raw, pk=request.POST['idSelected'])
        print('runModel fileRaw', fileRaw, fileRaw.id, type(fileRaw.id), request.session['idSelected'], type(request.session['idSelected']))

        # save to database
        mr = Model_Run(file_raw_id=int(request.POST['idSelected']), model_ran=request.POST['model_ran'], notes=request.POST['notes'], epochs=int(request.POST['epochs']), pub_date=timezone.now())
        mr.save()

        orderStep = 1
        fcList = ['filters','activation','activation_output','optimizer','loss','learning_rate','model_option','dropout','earlystop']
        for param in fcList:
            if param in request.POST:
                mrs = Model_Run_Steps(model_run_id=mr.id, order=orderStep, step=param, value=request.POST[param])
                mrs.save()
                orderStep += 1

        weightedAccuracy = 0
        countAccuracy = 0
        for method in fileRaw.methods.split(','):
            for nFeats in fileRaw.number_features.split(','):
                # save to database
                mrf = Model_Run_Features(model_run_id=mr.id,
                                         method=method,
                                         num_features=nFeats,
                                         pub_date=timezone.now())
                mrf.save()

                # get parameters for model to run
                parametersList = getParametersList({mr}, True)

                # run model
                accuracyScore = specificModel(fileRaw, method, nFeats, request.POST['model_ran'], int(mrf.id), parametersList, int(request.POST['epochs']), fileRaw.target_class, fileRaw.id)
                mrf.accuracy_score = accuracyScore
                mrf.save()
                weightedAccuracy += accuracyScore
                countAccuracy += 1

        # save weighted accuracy score
        if countAccuracy > 0:
            mr.weighted_accuracy = weightedAccuracy/countAccuracy
            mr.save()

        parametersList = getParametersList(None)

        return render(request, 'benchmark/run.html', {
            'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']),
            'model_run_list': Model_Run.objects.filter(file_raw__web_user__id=request.session['userid']).order_by('-pub_date'),
            'parameters_list': parametersList,
            'active':"Run",
        })
    parametersList = getParametersList(None)
    return render(request, 'benchmark/run.html', {
        'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']), 
        'model_run_list': Model_Run.objects.filter(file_raw__web_user__id=request.session['userid']).order_by('-pub_date'),
        'parameters_list': parametersList,
        'active':"Run",
    })

# put all parameters together to display easily
def getParametersList(modelsList, single=False):
    parametersList = {}
    if modelsList == None:
        modelsList = Model_Run.objects.all()
    for mr in modelsList:
        steps = None
        for mrs in Model_Run_Steps.objects.filter(model_run_id=mr.id).order_by('order'):
            if steps == None:
                if single:
                    parametersList[mrs.step] = mrs.value
                else:
                    steps = mrs.step + ':' + mrs.value
            else:
                if single:
                    parametersList[mrs.step] = mrs.value
                else:
                    steps = steps + ', ' + mrs.step + ':' + mrs.value
        if not single:
            parametersList[mr.id] = steps
    return parametersList

# get value from dictionary
@register.filter
def getDictValue(dict, key):
    return dict.get(key)

# set variable=value
@register.simple_tag
def setvar(val="None"):
    return val

# delete database records and files pca, loss and accuracy
def deleteTableFiles(deleteFile, deleteModel, deleteFeature, dir):

    print('delete:', deleteFile, deleteModel, deleteFeature)
    if deleteFeature != None:
        # deleting accuracy and loss files for the model run features id
        for feature in Model_Run_Features.objects.filter(id=deleteFeature):
            for f in glob.glob(dir + str(feature.id) + "_accuracy.png"):
                print('deleting accuracy file ' + f)
                os.remove(f)
            for f in glob.glob(dir + str(feature.id) + "_loss.png"):
                print('deleting loss file ' + f)
                os.remove(f)

        # deleting feature id and cascade
        Model_Run_Features.objects.filter(id=deleteFeature).delete()

    if deleteModel != None:
        # deleting accuracy and loss files for the model run features id
        for feature in Model_Run_Features.objects.filter(model_run_id=deleteModel):
            for f in glob.glob(dir + str(feature.id) + "_accuracy.png"):
                print('deleting accuracy file ' + f)
                os.remove(f)
            for f in glob.glob(dir + str(feature.id) + "_loss.png"):
                print('deleting loss file ' + f)
                os.remove(f)

        # deleting feature id and cascade
        Model_Run.objects.filter(id=deleteModel).delete()

    if deleteFile != None:
        # deleting accuracy and loss files for the model run features id
        for feature in Model_Run_Features.objects.filter(model_run__file_raw__id=deleteFile):
            for f in glob.glob(dir + str(feature.id) + "_accuracy.png"):
                print('deleting accuracy file ' + f)
                os.remove(f)
            for f in glob.glob(dir + str(feature.id) + "_loss.png"):
                print('deleting loss file ' + f)
                os.remove(f)

        # deleting pca files for the file raw id
        print('delete file raw table and files ', deleteFile)
        for f in glob.glob(dir + deleteFile + "_*.pca.png"):
            print('deleting pca file ' + f)
            os.remove(f)

        # deleting file raw id and cascade
        File_raw.objects.filter(id=deleteFile).delete()

# evaluate ML model
def evaluate(request):
    if 'userid' not in request.session or request.session['userid'] == None:
        return render(request, 'benchmark/signin.html', {
            'message':'Please enter username and password',
            'active':"Signin",
        })
    
    # delete tables rows and file for: feature or model or file and cascade
    if request.method == 'POST':
        if request.POST['deleteFile']:
            deleteTableFiles(request.POST['deleteFile'], None, None, 'benchmark/static/benchmark/images/')
        if request.POST['deleteModel']:
            deleteTableFiles(None, request.POST['deleteModel'], None, 'benchmark/static/benchmark/images/')
        if request.POST['deleteFeature']:
            deleteTableFiles(None, None, request.POST['deleteFeature'], 'benchmark/static/benchmark/images/')

    if request.method == 'POST' and request.POST['radioSelection']:
        print('evaluate runModel radio selection', request.POST['radioSelection'])
        fileRaw = get_object_or_404(File_raw, pk=request.POST['idSelected'])
        print('runModel fileRaw', fileRaw)
        modelsList = Model_Run.objects.filter(file_raw_id=request.POST['idSelected']).order_by('-pub_date')
        parametersList = getParametersList(modelsList)
        idModelRunFeatures = None
        classificationReportList = None
        confusionMatrixList = None
        accuracyCurve = None
        lossCurve = None
        modelFeaturesList = None

        if 'id_model_run' in request.POST:
            idModelRun = int(request.POST['id_model_run'])
            print('id model run', idModelRun)
            if request.POST['radioSelection'] in ('2','3'):
                print('request radio selection if', request.POST['radioSelection'])
                modelFeaturesList = Model_Run_Features.objects.filter(model_run_id=idModelRun).order_by('-pub_date')
            else:
                print('request radio selection', request.POST['radioSelection'])
            if 'id_model_run_features' in request.POST:
                idModelRunFeatures = int(request.POST['id_model_run_features'])
                print('idModelRunFeatures', idModelRunFeatures)

                if request.POST['radioSelection'] in ('3'):
                    classificationReportList = Classification_Report.objects.filter(model_run_features_id=idModelRunFeatures)
                    confusionMatrixList = Confusion_Matrix.objects.filter(model_run_features_id=idModelRunFeatures)
                    accuracyCurve = '/static/benchmark/images/'+str(idModelRunFeatures)+'_accuracy.png'
                    lossCurve = '/static/benchmark/images/'+str(idModelRunFeatures)+'_loss.png'
        else:
            idModelRun = None
            print('id model run', idModelRun)

        return render(request, 'benchmark/evaluate.html', {
            'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']),
            'model_run_list': modelsList,
            'model_run_features_list': modelFeaturesList,
            'idSelected': int(request.POST['idSelected']),
            'idModelRun': idModelRun,
            'parametersList': parametersList,
            'idModelRunFeatures': idModelRunFeatures,
            'classification_report_list': classificationReportList,
            'confusion_matrix_list': confusionMatrixList,
            'accuracy_curve': accuracyCurve,
            'loss_curve': lossCurve,
            'active':"Evaluate",
        })
    return render(request, 'benchmark/evaluate.html', {
        'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']), 
        'active':"Evaluate",
    })

# sign in/login
def signin(request):
    message = None
    if request.method == 'POST' and request.POST['username']:
        print('user sign in', request.POST['username'])
        try:
            webUser = Web_User.objects.get(username=request.POST['username'])
        except Web_User.DoesNotExist:
            webUser = None

        if webUser == None:
            message = 'wrong username or password'
            print('no web user', message)
            request.session['userid'] = None
            logout(request)
        else:
            print('web user')
            if 'password' in request.POST and request.POST['password'] == webUser.password:
                request.session['userid'] = webUser.id
                return render(request, 'benchmark/score.html', {
                    'file_raw_list': File_raw.objects.filter(web_user_id=webUser.id).order_by('-pub_date'),
                    'active':"Load",
                })
            else:
                message = 'wrong username or password'
                print('password no match', message)
                request.session['userid'] = None
                logout(request)
    else:
        message = 'Please enter username and password'
        request.session['userid'] = None
        logout(request)

    return render(request, 'benchmark/signin.html', {
        'message':message,
        'active':"Signin",
    })

# sign out / logout
def signout(request):

    message = None
    request.session['userid'] = None

    return render(request, 'benchmark/index.html', {
        'active':"Home",
    })

# compare ML models
def compare(request):
    if 'userid' not in request.session or request.session['userid'] == None:
        return render(request, 'benchmark/signin.html', {
            'message':'Please enter username and password',
            'active':"Signin",
        })
    
    if request.method == 'POST' and request.POST['radioSelection']:
        print('evaluate runModel radio selection', request.POST['radioSelection'])
        fileRaw = get_object_or_404(File_raw, pk=request.POST['idSelected'])
        print('runModel fileRaw', fileRaw)
        modelsList = Model_Run.objects.filter(file_raw_id=request.POST['idSelected']).order_by('-pub_date')
        parametersList = getParametersList(modelsList)
        idModelRunFeatures = None
        classificationReportList = None
        confusionMatrixList = None
        accuracyCurve = None
        lossCurve = None
        modelFeaturesList = None

        if 'id_model_run' in request.POST:
            idModelRun = int(request.POST['id_model_run'])
            print('id model run', idModelRun)
            if request.POST['radioSelection'] in ('2','3'):
                print('request radio selection if', request.POST['radioSelection'])
                modelFeaturesList = Model_Run_Features.objects.filter(model_run_id=idModelRun).order_by('-pub_date')
            else:
                print('request radio selection', request.POST['radioSelection'])
            if 'id_model_run_features' in request.POST:
                idModelRunFeatures = int(request.POST['id_model_run_features'])
                print('idModelRunFeatures', idModelRunFeatures)

                if request.POST['radioSelection'] in ('3'):
                    classificationReportList = Classification_Report.objects.filter(model_run_features_id=idModelRunFeatures)
                    confusionMatrixList = Confusion_Matrix.objects.filter(model_run_features_id=idModelRunFeatures)
                    accuracyCurve = '/static/benchmark/images/'+str(idModelRunFeatures)+'_accuracy.png'
                    lossCurve = '/static/benchmark/images/'+str(idModelRunFeatures)+'_loss.png'
        else:
            idModelRun = None
            print('id model run', idModelRun)

        return render(request, 'benchmark/compare.html', {
            'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']),
            'model_run_list': modelsList,
            'model_run_features_list': modelFeaturesList,
            'idSelected': int(request.POST['idSelected']),
            'idModelRun': idModelRun,
            'parametersList': parametersList,
            'idModelRunFeatures': idModelRunFeatures,
            'classification_report_list': classificationReportList,
            'confusion_matrix_list': confusionMatrixList,
            'accuracy_curve': accuracyCurve,
            'loss_curve': lossCurve,
            'active':"Compare",
        })
    return render(request, 'benchmark/compare.html', {
        'file_raw_list': File_raw.objects.filter(web_user_id=request.session['userid']), 
        'active':"Compare",
    })

def vote(request, question_id):
    question = get_object_or_404(Question, pk=question_id)
    try:
        selected_choice = question.choice_set.get(pk=request.POST['choice'])
    except (KeyError, Choice.DoesNotExist):
        # Redisplay the question voting form.
        return render(request, 'benchmark/detail.html', {
            'question': question,
            'error_message': "You didn't select a choice.",
        })
    else:
        selected_choice.votes += 1
        selected_choice.save()
        # Always return an HttpResponseRedirect after successfully dealing
        # with POST data. This prevents data from being posted twice if a
        # user hits the Back button.
        return HttpResponseRedirect(reverse('benchmark:results', args=(question.id,)))

