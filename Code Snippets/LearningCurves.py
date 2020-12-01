### Model Learning Curves
def plot_learning_curves(model):
  '''
  Learning Curves
  '''
  plt.grid()
  plt.plot(history.history['auc'], 'o-', color = 'red')
  plt.fill_between(list(range(epochs)), 
                   np.mean(history.history['auc']) - np.std(history.history['auc']),
                   np.mean(history.history['auc']) + np.std(history.history['auc']), 
                   alpha=0.1,
                   color="r")
  plt.plot(history.history['val_auc'], 'o-', color = 'blue')
  plt.fill_between(list(range(epochs)), 
                   np.mean(history.history['val_auc']) - np.std(history.history['val_auc']),
                   np.mean(history.history['val_auc']) + np.std(history.history['val_auc']), 
                   alpha=0.1,
                   color="b")
  plt.title('Model ROC AUC Scores')
  plt.ylabel('AUC Score')
  plt.xlabel('Epochs')
  plt.legend(['train', 'valid'], loc='bottom right')
  plt.savefig(f"Learning_Curves_{model}.png")
  plt.show()


### Model Loss Curves
def plot_loss_curves(model):
  '''
  Loss Curves
  '''
  plt.grid()
  plt.plot(history.history['loss'], 'o-', color = 'purple')
  plt.fill_between(list(range(epochs)), 
                  np.mean(history.history['loss']) - np.std(history.history['loss']),
                  np.mean(history.history['loss']) + np.std(history.history['loss']), 
                  alpha=0.1,
                  color='purple')
  plt.plot(history.history['val_loss'], 'o-', color = 'deepskyblue')
  plt.fill_between(list(range(epochs)), 
                  np.mean(history.history['val_loss']) - np.std(history.history['val_loss']),
                  np.mean(history.history['val_loss']) + np.std(history.history['val_loss']), 
                  alpha=0.1,
                  color="deepskyblue")
  plt.title('Model Loss Curves')
  plt.ylabel('Loss')
  plt.xlabel('Epochs')
  plt.legend(['train', 'valid'], loc='upper right')
  plt.savefig(f"Loss_Curves_{model}.png")
  plt.show()

### ROC AUC Curve
from sklearn.metrics import roc_curve
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer

def multiclass_roc_auc_score(y_test, y_pred, y_pred_score, model_name, average="macro"):
    '''
    Plotting ROC-AUC Curves
    '''        
    fpr, tpr, _ = roc_curve(y_test, y_pred_score)
    no_score_probs = [0 for _ in range(len(y_test))]
    ns_fpr, ns_tpr, _ = roc_curve(y_test, no_score_probs)

    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    '''
    Plotting ROC Curves
    '''
    plt.grid()
    plt.plot(fpr, tpr, marker='.', label='ROC Curve', color='darkorange')
    plt.plot(ns_fpr, ns_tpr, linestyle='--', color='darkblue')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves')
    # show the legend
    plt.legend()
    # show the plot
    plt.savefig(f"ROC_AUC_Curve_{model_name}.png")
    plt.show()
