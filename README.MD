```python
import pandas as pd
import seaborn as sns
import numpy as np
import panel as pn
pn.extension('tabulator')
import hvplot.pandas
from panel.interact import interact
```






<div id='45d28f12-ce8b-4db4-a75c-043694581d03'>
  <div id="efc2cac1-f6e8-4529-ba05-cc45c5908277" data-root-id="45d28f12-ce8b-4db4-a75c-043694581d03" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"fe33d2e6-0f9d-4574-a0f2-405f4d64dac0":{"version":"3.1.1","title":"Bokeh Application","defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}],"roots":[{"type":"object","name":"panel.models.browser.BrowserInfo","id":"45d28f12-ce8b-4db4-a75c-043694581d03"},{"type":"object","name":"panel.models.comm_manager.CommManager","id":"7003ccd2-6fa8-42b6-8966-738f8866dbc7","attributes":{"plot_id":"45d28f12-ce8b-4db4-a75c-043694581d03","comm_id":"7972de59ec284ff79984a1f381106593","client_comm_id":"73fc262e2ed34efd8302d117d49eb5a8"}}],"callbacks":{"type":"map"}}};
  var render_items = [{"docid":"fe33d2e6-0f9d-4574-a0f2-405f4d64dac0","roots":{"45d28f12-ce8b-4db4-a75c-043694581d03":"efc2cac1-f6e8-4529-ba05-cc45c5908277"},"root_ids":["45d28f12-ce8b-4db4-a75c-043694581d03"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined && ( root['Tabulator'] !== undefined) && ( root['Tabulator'] !== undefined))
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>







<style>*[data-root-id],
*[data-root-id] > * {
  box-sizing: border-box;
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  color: var(--vscode-editor-foreground, var(--jp-ui-font-color1));
}

/* Override VSCode background color */
.cell-output-ipywidget-background:has(> .cell-output-ipywidget-background
    > .lm-Widget
    > *[data-root-id]),
.cell-output-ipywidget-background:has(> .lm-Widget > *[data-root-id]) {
  background-color: transparent !important;
}
</style>



```python
df = pd.read_csv(
    "data/penguins_wNames.csv"
)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 333 entries, 0 to 332
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   species            333 non-null    object 
     1   island             333 non-null    object 
     2   bill_length_mm     333 non-null    float64
     3   bill_depth_mm      333 non-null    float64
     4   flipper_length_mm  333 non-null    float64
     5   body_mass_g        333 non-null    float64
     6   sex                333 non-null    object 
     7   year               333 non-null    int64  
     8   name               333 non-null    object 
    dtypes: float64(4), int64(1), object(4)
    memory usage: 23.5+ KB
    


```python
# NAs across the dataset
df.apply(lambda x: sum(x.isnull().values), axis = 0)
```




    species              0
    island               0
    bill_length_mm       0
    bill_depth_mm        0
    flipper_length_mm    0
    body_mass_g          0
    sex                  0
    year                 0
    name                 0
    dtype: int32




```python
df = df.dropna()
```


```python
df.hvplot.scatter(x='bill_length_mm', y='bill_depth_mm', by='species')
```






<div id='74c3359f-bee9-43f8-ae04-440a17363e08'>
  <div id="e22865c4-01ba-4f1e-987d-059e31edaa94" data-root-id="74c3359f-bee9-43f8-ae04-440a17363e08" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"533a25d2-0e47-4e42-9a05-2239735f7626":{"version":"3.1.1","title":"Bokeh Application","defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}],"roots":[{"type":"object","name":"Row","id":"74c3359f-bee9-43f8-ae04-440a17363e08","attributes":{"name":"Row46190","tags":["embedded"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"6870017e-7f35-43a9-91e4-f2e5f91e873f","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"017a010f-65b5-40e7-8307-dc77624d9009","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"2ebee49a-4126-4a21-b460-097d599baf8e","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"06981fc0-7bd5-4133-a064-f200c0e48398","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/native.css"}}],"min_width":0,"margin":0,"sizing_mode":"stretch_width","align":"start","children":[{"type":"object","name":"Spacer","id":"f3cea5a5-b39e-4d07-ae27-ed1ed1f1678c","attributes":{"name":"HSpacer46201","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"6870017e-7f35-43a9-91e4-f2e5f91e873f"},{"id":"2ebee49a-4126-4a21-b460-097d599baf8e"},{"id":"06981fc0-7bd5-4133-a064-f200c0e48398"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}},{"type":"object","name":"Figure","id":"7f1fc6f9-65ea-4427-b510-2c05aa573783","attributes":{"width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"19e38511-4a7a-4e2a-8d45-37780bc622e7","attributes":{"tags":[[["bill_length_mm","bill_length_mm",null]]],"start":30.921428571428574,"end":60.77857142857143,"reset_start":30.921428571428574,"reset_end":60.77857142857143}},"y_range":{"type":"object","name":"Range1d","id":"f181aab2-c2b9-4eb9-9ae5-56b7dd6eaeb5","attributes":{"tags":[[["bill_depth_mm","bill_depth_mm",null]]],"start":12.26,"end":22.34,"reset_start":12.26,"reset_end":22.34}},"x_scale":{"type":"object","name":"LinearScale","id":"7062eb7b-6e73-4ca7-ad07-89cca5c4fcac"},"y_scale":{"type":"object","name":"LinearScale","id":"fc7e73d1-ef53-4927-80d7-ef4b0e4016ba"},"title":{"type":"object","name":"Title","id":"4a626425-557e-413a-9797-d1a8c4d629ec","attributes":{"text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"4b9861a8-d963-444e-82bf-b189096e9cc7","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"066bf939-24b1-402c-8d96-680acc230840","attributes":{"selected":{"type":"object","name":"Selection","id":"756e456e-1697-4a3d-9c42-afc8bc6a56e0","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"b9a9bb75-1b5a-4536-a624-828896878c09"},"data":{"type":"map","entries":[["bill_length_mm",{"type":"ndarray","array":{"type":"bytes","data":"zczMzMyMQ0AAAAAAAMBDQGZmZmZmJkRAmpmZmZlZQkBmZmZmZqZDQDMzMzMzc0NAmpmZmZmZQ0DNzMzMzIxEQM3MzMzMTENAzczMzMxMQUDNzMzMzExCQJqZmZmZWUNAAAAAAABARUAzMzMzMzNBQAAAAAAAAEdAZmZmZmbmQkCamZmZmdlCQDMzMzMz80FAmpmZmZkZQ0BmZmZmZmZDQGZmZmZmpkFAzczMzMxMREAAAAAAAEBEQDMzMzMz80JAAAAAAABAREAAAAAAAMBDQJqZmZmZmUJAAAAAAADAQ0AzMzMzM3NEQDMzMzMzM0JAmpmZmZmZQ0BmZmZmZmZDQJqZmZmZGUVAzczMzMzMQkBmZmZmZuZDQAAAAAAAQEJAZmZmZmZmREAAAAAAAABCQM3MzMzMDEZAAAAAAACAQkDNzMzMzMxDQM3MzMzMjERAAAAAAAAAQkBmZmZmZiZFQM3MzMzMzENAzczMzMwMREAAAAAAAIBBQAAAAAAAAEVAAAAAAABAQUAzMzMzM7NEQAAAAAAAgENAzczMzMxMREAAAAAAAEBCQM3MzMzMzEJAmpmZmZnZQUBmZmZmZqZEQM3MzMzMzEJAzczMzMyMREAzMzMzMzNCQM3MzMzMzERAAAAAAADAQUDNzMzMzIxEQDMzMzMz80FAZmZmZmbmREAAAAAAAMBAQJqZmZmZ2UNAzczMzMzMQ0BmZmZmZuZGQAAAAAAAwEFAZmZmZmZmRUAzMzMzM3NEQJqZmZmZmUJAmpmZmZkZQkDNzMzMzAxFQM3MzMzMTEFAMzMzMzNzRUCamZmZmVlCQM3MzMzMjEFAZmZmZmamQkBmZmZmZqZEQGZmZmZmJkJAMzMzMzNzQkBmZmZmZiZDQDMzMzMzc0NAmpmZmZnZQUDNzMzMzIxEQAAAAAAAAEFAzczMzMzMQ0CamZmZmRlCQGZmZmZmZkRAzczMzMwMQ0BmZmZmZiZEQM3MzMzMjEBAmpmZmZmZRUAAAAAAAIBBQAAAAAAAgERAmpmZmZnZQkBmZmZmZuZCQDMzMzMz80JAmpmZmZnZQ0DNzMzMzExDQJqZmZmZGUNAzczMzMwMQ0CamZmZmZlFQM3MzMzMDENAzczMzMzMRkCamZmZmdlDQJqZmZmZGUVAzczMzMzMQ0CamZmZmVlFQM3MzMzMTENAZmZmZmamQkCamZmZmdlBQM3MzMzMjERAmpmZmZkZQkCamZmZmdlCQJqZmZmZGURAMzMzMzOzRECamZmZmZlBQM3MzMzMTERAZmZmZmZmQ0AAAAAAAMBEQAAAAAAAgENAzczMzMwMRkAAAAAAAEBDQM3MzMzMjEVAZmZmZmZmQkAAAAAAAMBCQM3MzMzMDENAzczMzMyMREDNzMzMzMxBQJqZmZmZGURAAAAAAACAQkCamZmZmdlDQJqZmZmZGURAzczMzMxMREDNzMzMzAxAQJqZmZmZWURAZmZmZmamQkAAAAAAAIBDQJqZmZmZmUNAzczMzMxMQkAAAAAAAABCQGZmZmZm5kJAAAAAAAAAQkAAAAAAAMBEQA=="},"shape":[146],"dtype":"float64","order":"little"}],["bill_depth_mm",{"type":"ndarray","array":{"type":"bytes","data":"MzMzMzOzMkBmZmZmZmYxQAAAAAAAADJAzczMzMxMM0CamZmZmZk0QM3MzMzMzDFAmpmZmZmZM0CamZmZmZkxQDMzMzMzMzVAmpmZmZkZNUDNzMzMzMwxQAAAAAAAADNAMzMzMzOzNEBmZmZmZmYyQAAAAAAAgDVAzczMzMxMMkAzMzMzM7MyQDMzMzMzMzNAmpmZmZkZMkAzMzMzMzMxQGZmZmZm5jJAmpmZmZmZMkBmZmZmZuYxQJqZmZmZmTJAZmZmZmbmMkAzMzMzM7MwQJqZmZmZGTJAzczMzMzMMUBmZmZmZuYyQAAAAAAAADFAmpmZmZkZNUAAAAAAAAA0QAAAAAAAgDJAzczMzMxMM0CamZmZmRkzQAAAAAAAADJAZmZmZmZmMkAAAAAAAIAyQDMzMzMzszNAZmZmZmbmMEDNzMzMzMwyQAAAAAAAADNAZmZmZmbmMUAzMzMzMzM1QDMzMzMzszFAZmZmZmbmMkBmZmZmZuYxQAAAAAAAgDNAmpmZmZkZMkCamZmZmZkyQAAAAAAAgDFAzczMzMzMMkCamZmZmZkwQJqZmZmZGTNAZmZmZmbmMECamZmZmRk1QAAAAAAAADFAMzMzMzMzMkCamZmZmRkxQAAAAAAAADJAMzMzMzMzMECamZmZmRkzQJqZmZmZmTBAZmZmZmZmM0AAAAAAAAAzQGZmZmZmZjJAMzMzMzMzMUBmZmZmZuYyQAAAAAAAgDFAAAAAAACAMkDNzMzMzMwwQGZmZmZmZjNAmpmZmZkZMECamZmZmRkzQDMzMzMzMzFAmpmZmZmZMUDNzMzMzMwyQGZmZmZmZjNAzczMzMzMMUDNzMzMzEw0QAAAAAAAgDNAmpmZmZmZMkAzMzMzMzMzQM3MzMzMzDJAAAAAAAAAMkCamZmZmRkyQJqZmZmZGTFAmpmZmZkZMkDNzMzMzEwxQGZmZmZm5jJAmpmZmZmZMkAAAAAAAIAyQJqZmZmZGTBAAAAAAACAMkBmZmZmZuYxQAAAAAAAADRAAAAAAAAAMEAAAAAAAAA0QJqZmZmZmTJAZmZmZmbmMkAzMzMzMzMxQAAAAAAAADRAAAAAAAAAMUAAAAAAAAAzQAAAAAAAgDBAzczMzMxMNEAzMzMzM7MxQAAAAAAAgDNAMzMzMzOzNEDNzMzMzEwyQAAAAAAAADFAAAAAAACANEAAAAAAAAAxQJqZmZmZmTJAMzMzMzMzMUDNzMzMzMwzQAAAAAAAADFAAAAAAACAMkDNzMzMzMwvQAAAAAAAADNAmpmZmZmZMUDNzMzMzEwyQJqZmZmZGTFAAAAAAAAAMkBmZmZmZuYxQDMzMzMzMzNAAAAAAACAMkAAAAAAAIAyQJqZmZmZmTFAAAAAAACAMUAAAAAAAIAxQJqZmZmZGTRAAAAAAACAMEBmZmZmZuYxQJqZmZmZGTFAMzMzMzMzMUAAAAAAAAAvQAAAAAAAADFAzczMzMzMMEAzMzMzM7MyQJqZmZmZmTJAZmZmZmZmMkDNzMzMzMwxQJqZmZmZGTJAmpmZmZkZMUAAAAAAAIAyQA=="},"shape":[146],"dtype":"float64","order":"little"}],["species",["Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie"]]]}}},"view":{"type":"object","name":"CDSView","id":"b79e7419-9f63-4b4e-bd57-c2bea48ce906","attributes":{"filter":{"type":"object","name":"AllIndices","id":"d31bce98-1ca1-45a4-86bc-f8b0bd20645a"}}},"glyph":{"type":"object","name":"Scatter","id":"42b3a92a-3a2e-4872-b305-db95b8d8051b","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"64676092-443e-4b45-8533-5c4441dbb511","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"e4c1164e-8b7b-4b97-9083-0a31254dfee4","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"5702942f-d999-4aed-8ee9-5a349789f9ac","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"041e823a-6f7b-4054-9665-75e8d7df02ea","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"bf512fc6-1616-4614-8fd3-a10b87416b3e","attributes":{"selected":{"type":"object","name":"Selection","id":"a45e4e96-8bb3-4ef6-afcd-ecf08e3d54a5","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"d898ddf8-d0ff-4a7f-8a3a-4071333c5558"},"data":{"type":"map","entries":[["bill_length_mm",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAABAR0AAAAAAAABJQGZmZmZmpklAMzMzMzOzRkCamZmZmVlKQJqZmZmZmUZAzczMzMwMR0BmZmZmZqZJQAAAAAAAAEdAZmZmZmamSUDNzMzMzExHQJqZmZmZ2UlAAAAAAACAR0AAAAAAAABKQDMzMzMz80ZAAAAAAABASUBmZmZmZiZJQAAAAAAAAE1AMzMzMzMzR0CamZmZmZlIQDMzMzMzM0VAAAAAAABASECamZmZmZlFQM3MzMzMTElAmpmZmZlZR0AAAAAAAABKQAAAAAAAQElAAAAAAADASEAzMzMzMzNHQGZmZmZmZkpAMzMzMzNzRECamZmZmRlLQAAAAAAAQEVAAAAAAACASUCamZmZmdlIQAAAAAAAwEdAzczMzMzMR0AAAAAAAABKQDMzMzMzc0dAAAAAAADASkAAAAAAAIBIQJqZmZmZGUdAMzMzMzNzSUAAAAAAAMBGQDMzMzMzc0lAZmZmZmZmSUDNzMzMzAxJQAAAAAAAgEhAAAAAAADASUBmZmZmZuZIQM3MzMzMDEhAMzMzMzOzSUCamZmZmdlGQJqZmZmZWUlAAAAAAABARUCamZmZmRlKQJqZmZmZmUZAZmZmZmamSECamZmZmRlJQM3MzMzMzEZAMzMzMzPzSUBmZmZmZmZHQJqZmZmZ2UZAZmZmZmbmS0AAAAAAAMBFQM3MzMzMzEhAZmZmZmZmSUCamZmZmRlJQA=="},"shape":[68],"dtype":"float64","order":"little"}],["bill_depth_mm",{"type":"ndarray","array":{"type":"bytes","data":"ZmZmZmbmMUAAAAAAAIAzQDMzMzMzMzNAMzMzMzOzMkDNzMzMzMwzQM3MzMzMzDFAMzMzMzMzMkAzMzMzMzMyQGZmZmZm5jJAZmZmZmbmM0DNzMzMzMwxQM3MzMzMTDRAzczMzMxMMUCamZmZmRkyQJqZmZmZGTFAmpmZmZmZM0AAAAAAAAA0QM3MzMzMzDFAmpmZmZmZMkAzMzMzMzMyQM3MzMzMTDFAAAAAAACAMUCamZmZmZkwQGZmZmZmZjNAZmZmZmbmMUAAAAAAAAAzQGZmZmZmZjJAAAAAAAAAM0DNzMzMzMwxQAAAAAAAADRAmpmZmZmZMEDNzMzMzMw0QDMzMzMzszBAzczMzMzMMkCamZmZmZkyQM3MzMzMzDBAzczMzMxMMkAzMzMzM7M0QJqZmZmZmTBAZmZmZmbmM0AAAAAAAIAzQAAAAAAAgDFAmpmZmZkZM0AAAAAAAAAxQGZmZmZm5jFAAAAAAACAMkBmZmZmZuYxQJqZmZmZmTNAMzMzMzOzMkDNzMzMzEwxQGZmZmZmZjBAAAAAAAAAM0DNzMzMzEwxQDMzMzMzszNAzczMzMxMMUDNzMzMzMwyQJqZmZmZmTBAZmZmZmbmM0DNzMzMzMwyQGZmZmZmZjNAAAAAAACAM0AAAAAAAIAwQAAAAAAAADFAzczMzMzMM0CamZmZmRkyQDMzMzMzMzJAAAAAAAAAM0AzMzMzM7MyQA=="},"shape":[68],"dtype":"float64","order":"little"}],["species",["Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap","Chinstrap"]]]}}},"view":{"type":"object","name":"CDSView","id":"b120a468-d4a2-4ce9-9942-7e3c895f152c","attributes":{"filter":{"type":"object","name":"AllIndices","id":"c15520d2-80b3-4fe6-9b09-abc33eb12184"}}},"glyph":{"type":"object","name":"Scatter","id":"25980fc0-5957-4e26-a090-e86179d5cfcc","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"fill_color":{"type":"value","value":"#fc4f30"},"hatch_color":{"type":"value","value":"#fc4f30"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"9fe39304-8bbb-450e-917b-670aed75b6b3","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"6597de4b-59d3-421c-bab8-5951a68dce96","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"37f9e055-7f08-4109-9371-e072cd8be22d","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#fc4f30"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#fc4f30"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#fc4f30"},"hatch_alpha":{"type":"value","value":0.2}}}}},{"type":"object","name":"GlyphRenderer","id":"1b7e7e44-3c3f-4f30-ae13-7472fcc203b6","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"143632a9-98ac-4ef3-b2cd-c7a9982c99bf","attributes":{"selected":{"type":"object","name":"Selection","id":"1fb90daf-6e67-4c8b-859d-e24aafcab170","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"d590e78e-5485-4cb3-866b-cdf7fd405cb3"},"data":{"type":"map","entries":[["bill_length_mm",{"type":"ndarray","array":{"type":"bytes","data":"zczMzMwMR0AAAAAAAABJQJqZmZmZWUhAAAAAAAAASUDNzMzMzMxHQAAAAAAAQEdAMzMzMzOzRkCamZmZmVlHQGZmZmZmpkVAZmZmZmZmR0AzMzMzM3NEQAAAAAAAgEhAAAAAAADARkAzMzMzMzNIQGZmZmZm5kZAZmZmZmamSEAAAAAAAABFQJqZmZmZmUhAmpmZmZkZR0CamZmZmVlIQJqZmZmZGUlAzczMzMyMRkAAAAAAAEBHQGZmZmZmJkdAMzMzMzNzRUDNzMzMzAxHQGZmZmZm5kdAmpmZmZkZSEAAAAAAAABJQGZmZmZmpkdAZmZmZmZmRUDNzMzMzIxGQM3MzMzMzE1AzczMzMyMSEAzMzMzMzNIQM3MzMzMTEVAMzMzMzMzRkAAAAAAAABGQJqZmZmZWUhAmpmZmZlZRUDNzMzMzMxIQGZmZmZmpkZAzczMzMzMSEAAAAAAAEBJQM3MzMzMzEVAAAAAAADARkAAAAAAAEBJQDMzMzMzc0ZAmpmZmZmZRkDNzMzMzExHQAAAAAAAQEhAzczMzMyMRkDNzMzMzAxJQAAAAAAAQEdAAAAAAACARkBmZmZmZuZFQAAAAAAAwEZAmpmZmZmZRUAzMzMzMzNJQGZmZmZmpkZAmpmZmZkZR0CamZmZmdlGQGZmZmZmJktAZmZmZmbmRkBmZmZmZuZIQAAAAAAAwEhAAAAAAADARUCamZmZmVlJQJqZmZmZ2UdAMzMzMzMzR0CamZmZmRlIQAAAAAAAQEdAMzMzMzMzR0DNzMzMzExIQAAAAAAAwEdAzczMzMyMSUCamZmZmZlGQJqZmZmZmUZAzczMzMyMSEAAAAAAAEBKQDMzMzMzs0dAAAAAAAAASUAzMzMzM3NGQGZmZmZmZklAMzMzMzOzRUBmZmZmZqZJQAAAAAAAwEdAzczMzMwMSkAAAAAAAMBHQJqZmZmZGUpAAAAAAADARkAAAAAAAMBIQAAAAAAAQEZAZmZmZmZmSUAzMzMzM7NIQDMzMzMzc0dAMzMzMzMzSEDNzMzMzIxJQAAAAAAAQEhAMzMzMzPzS0CamZmZmZlHQM3MzMzMjEhAZmZmZmZmR0CamZmZmdlEQDMzMzMzs0pAZmZmZmamRUDNzMzMzAxIQAAAAAAAQElAZmZmZmbmSEAAAAAAAMBFQAAAAAAAwElAmpmZmZkZR0DNzMzMzIxLQGZmZmZmZkhAmpmZmZmZR0BmZmZmZmZHQDMzMzMzM0lAmpmZmZmZRkAzMzMzM/NIQA=="},"shape":[119],"dtype":"float64","order":"little"}],["bill_depth_mm",{"type":"ndarray","array":{"type":"bytes","data":"ZmZmZmZmKkDNzMzMzEwwQDMzMzMzMyxAZmZmZmZmLkAAAAAAAAAtQAAAAAAAACtAMzMzMzMzLUCamZmZmZkuQM3MzMzMzCpAzczMzMzMLkBmZmZmZmYrQJqZmZmZGTBAZmZmZmZmK0AzMzMzMzMtQDMzMzMzMy1AZmZmZmZmL0AAAAAAAAArQGZmZmZmZi5AAAAAAAAALUAzMzMzMzMuQJqZmZmZmSxAAAAAAAAALUAAAAAAAAAtQJqZmZmZmS9AMzMzMzMzKkAzMzMzMzMuQAAAAAAAAC5AmpmZmZmZLECamZmZmZkuQJqZmZmZmS5AZmZmZmZmLEAAAAAAAAAtQAAAAAAAADFAmpmZmZmZLUDNzMzMzEwwQGZmZmZmZitAzczMzMxMMUAzMzMzMzMrQGZmZmZmZi9AZmZmZmZmK0AAAAAAAAAwQGZmZmZmZitAAAAAAAAALkDNzMzMzMwvQM3MzMzMzCtAzczMzMzMK0DNzMzMzMwvQJqZmZmZmSpAmpmZmZmZL0BmZmZmZmYsQDMzMzMzMyxAzczMzMzMLEAAAAAAAAAuQM3MzMzMzCxAzczMzMzMLkDNzMzMzMwrQAAAAAAAAC5AAAAAAAAALUCamZmZmZkuQJqZmZmZmStAzczMzMzMLUDNzMzMzMwrQGZmZmZmZi9AZmZmZmZmLEDNzMzMzMwwQDMzMzMzMzBAZmZmZmZmLEAAAAAAAAAuQAAAAAAAAC5AMzMzMzMzL0AzMzMzMzMvQJqZmZmZmS1AAAAAAAAALkAAAAAAAAAwQGZmZmZmZixAzczMzMxMMECamZmZmZkrQGZmZmZmZjBAAAAAAAAALUAzMzMzMzMvQDMzMzMzMy1AzczMzMzML0CamZmZmZkrQM3MzMzMTDFAzczMzMzMLEBmZmZmZmYsQAAAAAAAACxAAAAAAAAAMUAAAAAAAAAuQJqZmZmZGTFAAAAAAAAALUCamZmZmRkwQGZmZmZmZi1AZmZmZmZmL0CamZmZmZkvQDMzMzMzMy1AzczMzMzMLEAAAAAAAIAwQAAAAAAAAC5AAAAAAAAAMUAAAAAAAAAvQAAAAAAAAC5AmpmZmZkZMEBmZmZmZmYtQJqZmZmZmS9AAAAAAAAALEAzMzMzMzMuQGZmZmZmZi5AzczMzMzML0BmZmZmZmYuQM3MzMzMTDBAMzMzMzMzLEAAAAAAAAAwQDMzMzMzMzBAZmZmZmZmK0CamZmZmZksQGZmZmZmZi9AmpmZmZmZLUCamZmZmRkwQA=="},"shape":[119],"dtype":"float64","order":"little"}],["species",["Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo","Gentoo"]]]}}},"view":{"type":"object","name":"CDSView","id":"8e1a2a40-93da-4639-bf17-4ea4a6e689da","attributes":{"filter":{"type":"object","name":"AllIndices","id":"777479d5-a160-420e-9b28-b406d1bfc59a"}}},"glyph":{"type":"object","name":"Scatter","id":"751fdd4e-adb6-426c-9071-b53880c63a25","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"fill_color":{"type":"value","value":"#e5ae38"},"hatch_color":{"type":"value","value":"#e5ae38"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"e084d1cf-9ccb-44a5-96ec-0973d272c906","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"2b9cd964-5f95-4b62-94b1-aafa38c445e8","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"a9b4542d-f568-4773-ae76-d13f9aede851","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"bill_length_mm"},"y":{"type":"field","field":"bill_depth_mm"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#e5ae38"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#e5ae38"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#e5ae38"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"b2d9c992-ef0d-482f-b22d-cb72706b983b","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"62e9f0db-b130-47f4-9de9-69b2d88c48ed","attributes":{"tags":["hv_created"],"renderers":[{"id":"4b9861a8-d963-444e-82bf-b189096e9cc7"},{"id":"041e823a-6f7b-4054-9665-75e8d7df02ea"},{"id":"1b7e7e44-3c3f-4f30-ae13-7472fcc203b6"}],"tooltips":[["species","@{species}"],["bill_length_mm","@{bill_length_mm}"],["bill_depth_mm","@{bill_depth_mm}"]]}},{"type":"object","name":"SaveTool","id":"541f45a3-9a05-4222-904e-1950459e0a53"},{"type":"object","name":"PanTool","id":"cfb56d7f-f777-4fb1-a19d-2efea648d1f3"},{"type":"object","name":"WheelZoomTool","id":"5d3d9016-f6a6-4a44-8332-6130b385992c"},{"type":"object","name":"BoxZoomTool","id":"01bd7622-631c-45ad-b1e6-0cbc9f0db365","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"c51c08bc-16ff-4f11-bbc0-c756530032af","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"c7ad320a-f9a1-4c90-9aa0-c158d2a46160"}],"active_drag":{"id":"cfb56d7f-f777-4fb1-a19d-2efea648d1f3"},"active_scroll":{"id":"5d3d9016-f6a6-4a44-8332-6130b385992c"}}},"left":[{"type":"object","name":"LinearAxis","id":"ed33b943-1c8f-4c85-b532-1acc452f08c9","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"308b823f-0650-4dcf-9b01-a43bd5a96888","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"888e7845-fca9-431c-8db6-5aac100435a2"},"axis_label":"bill_depth_mm","major_label_policy":{"type":"object","name":"AllLabels","id":"4c247efd-1ac8-49db-b664-5d02e845c836"}}}],"right":[{"type":"object","name":"Legend","id":"57ae7fa0-c49b-42c2-92b0-06ca14411e94","attributes":{"location":[0,0],"title":"species","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"54355c53-9c35-4e0b-ba70-3c6cc0f91f80","attributes":{"label":{"type":"value","value":"Adelie"},"renderers":[{"id":"4b9861a8-d963-444e-82bf-b189096e9cc7"}]}},{"type":"object","name":"LegendItem","id":"08c9719e-acf2-4ac3-8e9d-a42706b917d5","attributes":{"label":{"type":"value","value":"Chinstrap"},"renderers":[{"id":"041e823a-6f7b-4054-9665-75e8d7df02ea"}]}},{"type":"object","name":"LegendItem","id":"b34c02e5-4f20-4aab-bace-a0a4d1c96452","attributes":{"label":{"type":"value","value":"Gentoo"},"renderers":[{"id":"1b7e7e44-3c3f-4f30-ae13-7472fcc203b6"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"6b0163bb-73a1-4f34-b1e1-a1bf6fa9ac5a","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"7c9590d7-24a9-45ce-a8a9-19ebd0eb20ff","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"87cf14b1-554e-468e-9066-6d7d76548591"},"axis_label":"bill_length_mm","major_label_policy":{"type":"object","name":"AllLabels","id":"91947afe-f957-4604-bf65-0af256b61360"}}}],"center":[{"type":"object","name":"Grid","id":"3e85edb5-939d-4d7a-be4a-cb2a12f79649","attributes":{"axis":{"id":"6b0163bb-73a1-4f34-b1e1-a1bf6fa9ac5a"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"00022d36-7919-46b9-8518-8f0cb9da31c7","attributes":{"dimension":1,"axis":{"id":"ed33b943-1c8f-4c85-b532-1acc452f08c9"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}},{"type":"object","name":"Spacer","id":"61f01779-c945-4a37-81a5-96ec57f486fe","attributes":{"name":"HSpacer46204","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"6870017e-7f35-43a9-91e4-f2e5f91e873f"},{"id":"2ebee49a-4126-4a21-b460-097d599baf8e"},{"id":"06981fc0-7bd5-4133-a064-f200c0e48398"}],"margin":0,"sizing_mode":"stretch_width","align":"start"}}]}}],"callbacks":{"type":"map"}}};
  var render_items = [{"docid":"533a25d2-0e47-4e42-9a05-2239735f7626","roots":{"74c3359f-bee9-43f8-ae04-440a17363e08":"e22865c4-01ba-4f1e-987d-059e31edaa94"},"root_ids":["74c3359f-bee9-43f8-ae04-440a17363e08"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined && ( root['Tabulator'] !== undefined) && ( root['Tabulator'] !== undefined))
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
df_nums = df.select_dtypes(include=np.number)
df_corr= df_nums.dropna().corr()
corrplot = sns.heatmap(
    df_corr,
    annot = False,
)
```


    
![png](output_6_0.png)
    



```python
idf = df.interactive()
```


```python
radiobuttons = {}
for item in ("sex", "island", "species", "year"):
    radiobuttons[item] = pn.widgets.RadioButtonGroup(
        name = "Y axis",
        options = list(df[item].unique()),
        button_type = "primary"
    )
```


```python
radiobuttons["species"]
```






<div id='5a34e30d-e2d0-46db-8dce-d62f698a8bb1'>
  <div id="d53e0c6d-4f10-479f-a0e8-f32634e54132" data-root-id="5a34e30d-e2d0-46db-8dce-d62f698a8bb1" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"94c572cb-6731-4dc1-92e4-81fa0cb6897a":{"version":"3.1.1","title":"Bokeh Application","defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}],"roots":[{"type":"object","name":"RadioButtonGroup","id":"5a34e30d-e2d0-46db-8dce-d62f698a8bb1","attributes":{"button_type":"primary","css_classes":["solid"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"9e7b489a-cdd1-48f0-8e94-73a859d7ff32","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"649ecc1e-0763-4890-9bde-aabc6482ddd3","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/button.css"}},{"type":"object","name":"ImportedStyleSheet","id":"648ce7fd-d200-45bf-9eab-0a608aa4f76b","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"58247e2a-4e81-4a8f-8a07-423c128f6a52","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/native.css"}}],"margin":[5,10],"align":"start","labels":["Adelie","Gentoo","Chinstrap"],"active":0}},{"type":"object","name":"panel.models.comm_manager.CommManager","id":"06bb3376-2ea6-4a91-8261-91697ecff9d7","attributes":{"plot_id":"5a34e30d-e2d0-46db-8dce-d62f698a8bb1","comm_id":"829219a4122b408a8316fac2aa133eed","client_comm_id":"af6fc180ae654cb18c95fdfc994cade3"}}],"callbacks":{"type":"map"}}};
  var render_items = [{"docid":"94c572cb-6731-4dc1-92e4-81fa0cb6897a","roots":{"5a34e30d-e2d0-46db-8dce-d62f698a8bb1":"d53e0c6d-4f10-479f-a0e8-f32634e54132"},"root_ids":["5a34e30d-e2d0-46db-8dce-d62f698a8bb1"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined && ( root['Tabulator'] !== undefined) && ( root['Tabulator'] !== undefined))
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
islands = df["island"].unique().tolist()

island_select = pn.widgets.Select(name="Select Island", options=islands)
island_select = pn.widgets.RadioButtonGroup(name="Select Island", options = islands, button_type = "primary")

def create_plot(island):
    return df[df["island"] == island].hvplot(y = "body_mass_g",
                                             x = "flipper_length_mm",
                                             kind = "scatter",
                                             by = "species"
                                            )

interact(create_plot, island=island_select)
```








<div id='37b483bd-505e-40ca-b373-3dd614b1254d'>
  <div id="aa23c53d-f013-4ded-8fba-4abe6c1aa13f" data-root-id="37b483bd-505e-40ca-b373-3dd614b1254d" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"577ee033-a386-408a-8cae-264ee6081fd7":{"version":"3.1.1","title":"Bokeh Application","defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}],"roots":[{"type":"object","name":"Column","id":"37b483bd-505e-40ca-b373-3dd614b1254d","attributes":{"name":"Column50737","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"985f0d68-deae-46e3-8189-a02d222e238a","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"32e2da48-8430-4ee1-9fab-abdddb759459","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"d2535011-a1da-4aac-82f3-6a4c20557db6","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"876c8d33-165d-4e98-82e9-a57cf2e320c3","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/native.css"}}],"margin":0,"sizing_mode":"fixed","align":"start","children":[{"type":"object","name":"Column","id":"eee5a02a-f9ee-4131-b00a-2d32028b7926","attributes":{"name":"Column50854","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"985f0d68-deae-46e3-8189-a02d222e238a"},{"id":"32e2da48-8430-4ee1-9fab-abdddb759459"},{"id":"d2535011-a1da-4aac-82f3-6a4c20557db6"},{"id":"876c8d33-165d-4e98-82e9-a57cf2e320c3"}],"margin":0,"align":"start","children":[{"type":"object","name":"RadioButtonGroup","id":"3b9064b3-e2d1-480d-b0b3-301a5bd6c826","attributes":{"button_type":"primary","css_classes":["solid"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"985f0d68-deae-46e3-8189-a02d222e238a"},{"type":"object","name":"ImportedStyleSheet","id":"c5b1e604-945a-45da-a49c-2fba0ac74465","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/button.css"}},{"id":"d2535011-a1da-4aac-82f3-6a4c20557db6"},{"id":"876c8d33-165d-4e98-82e9-a57cf2e320c3"}],"margin":[5,10],"align":"start","labels":["Torgersen","Biscoe","Dream"],"active":0}}]}},{"type":"object","name":"Row","id":"da1312bf-8a3d-4700-9c35-f1280a616549","attributes":{"name":"Row50851","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"985f0d68-deae-46e3-8189-a02d222e238a"},{"id":"32e2da48-8430-4ee1-9fab-abdddb759459"},{"id":"d2535011-a1da-4aac-82f3-6a4c20557db6"},{"id":"876c8d33-165d-4e98-82e9-a57cf2e320c3"}],"margin":0,"sizing_mode":"fixed","align":"start","children":[{"type":"object","name":"Figure","id":"9f16f979-b8f4-478a-99ac-d4d60480925c","attributes":{"name":"interactive50734","width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"3d16d8ee-363b-4809-a885-8b4d47aa9b19","attributes":{"tags":[[["flipper_length_mm","flipper_length_mm",null]]],"start":174.54285714285714,"end":211.45714285714286,"reset_start":174.54285714285714,"reset_end":211.45714285714286}},"y_range":{"type":"object","name":"Range1d","id":"889a51da-2fa2-415e-bb9a-0db711b5755a","attributes":{"tags":[[["body_mass_g","body_mass_g",null]]],"start":2720.0,"end":4880.0,"reset_start":2720.0,"reset_end":4880.0}},"x_scale":{"type":"object","name":"LinearScale","id":"ec045705-ec87-4b12-8deb-eb211aa644bd"},"y_scale":{"type":"object","name":"LinearScale","id":"7995a340-e2a4-43e8-a259-c7e961318f85"},"title":{"type":"object","name":"Title","id":"b64693c3-24cf-4bce-bc40-8d876b758038","attributes":{"text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"ac5edb85-3e50-4fe4-a935-cc309206888d","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"d6c4a1b9-cca8-466c-984f-3140b22ab5a3","attributes":{"selected":{"type":"object","name":"Selection","id":"ebbc9e61-91d4-46b3-9a76-e9ab2d9c64c6","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"0813e00d-6a1b-47a4-a081-4d0db9c186bc"},"data":{"type":"map","entries":[["flipper_length_mm",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAACgZkAAAAAAAEBnQAAAAAAAYGhAAAAAAAAgaEAAAAAAAMBnQAAAAAAAoGZAAAAAAABgaEAAAAAAAMBmQAAAAAAA4GdAAAAAAADAaEAAAAAAACBnQAAAAAAAYGhAAAAAAACgaEAAAAAAAABnQAAAAAAAQGhAAAAAAADAZ0AAAAAAAMBoQAAAAAAAwGdAAAAAAADAZ0AAAAAAAIBoQAAAAAAAoGhAAAAAAADAZ0AAAAAAAGBoQAAAAAAA4GdAAAAAAAAAZ0AAAAAAAGBnQAAAAAAAYGhAAAAAAACgZ0AAAAAAAIBoQAAAAAAAYGdAAAAAAAAgaEAAAAAAAIBnQAAAAAAA4GhAAAAAAACgZ0AAAAAAAKBnQAAAAAAAYGdAAAAAAADAaEAAAAAAAABmQAAAAAAAQGlAAAAAAABAZ0AAAAAAAOBoQAAAAAAA4GdAAAAAAABgaEAAAAAAAOBnQAAAAAAAQGpAAAAAAADAZ0AAAAAAAKBoQA=="},"shape":[47],"dtype":"float64","order":"little"}],["body_mass_g",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAABMrUAAAAAAALCtQAAAAAAAZKlAAAAAAAD0qkAAAAAAAISsQAAAAAAAUqxAAAAAAABDskAAAAAAAACpQAAAAAAAsK1AAAAAAAAwsUAAAAAAAOisQAAAAAAA9KpAAAAAAACUsUAAAAAAAPqpQAAAAAAAaLBAAAAAAADUp0AAAAAAAGKxQAAAAAAAIKxAAAAAAAB4rkAAAAAAALyrQAAAAAAANrBAAAAAAADorEAAAAAAAJqwQAAAAAAA6KxAAAAAAAB4rkAAAAAAALyrQAAAAAAAQK9AAAAAAAAAqUAAAAAAAFyyQAAAAAAAsK1AAAAAAABosEAAAAAAAKimQAAAAAAAfq1AAAAAAAAsqkAAAAAAAPqpQAAAAAAAnKhAAAAAAABYq0AAAAAAAPSqQAAAAAAARq5AAAAAAADUp0AAAAAAAECvQAAAAAAAlqlAAAAAAADMsEAAAAAAANSnQAAAAAAAQK9AAAAAAAD6qUAAAAAAAFirQA=="},"shape":[47],"dtype":"float64","order":"little"}],["species",["Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie"]]]}}},"view":{"type":"object","name":"CDSView","id":"d30d1d99-b52e-4b32-92d1-373d671a59ff","attributes":{"filter":{"type":"object","name":"AllIndices","id":"aab4eda4-6499-4163-b476-133f5ccf4ee2"}}},"glyph":{"type":"object","name":"Scatter","id":"40ccb95f-ce92-4109-8082-94c55d047144","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"434aba54-138d-44b0-b6df-ac847eda5ab7","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"07c1dc23-5452-4be8-ac64-8237f9feac90","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"c7633a34-7245-408f-acd0-b6aee00a2877","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"62a814a7-ac4b-4331-b3a1-d5f75702c5e9","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"8cc2c7aa-4198-40ab-9228-3ccde48718a3","attributes":{"tags":["hv_created"],"renderers":[{"id":"ac5edb85-3e50-4fe4-a935-cc309206888d"}],"tooltips":[["species","@{species}"],["flipper_length_mm","@{flipper_length_mm}"],["body_mass_g","@{body_mass_g}"]]}},{"type":"object","name":"SaveTool","id":"c7ecdec7-88bf-4b43-83e6-7ced2a198794"},{"type":"object","name":"PanTool","id":"b1a927e1-14eb-461a-9abb-ff92507f4403"},{"type":"object","name":"WheelZoomTool","id":"d05b9d44-c600-422d-b8a0-cfc16a21d709"},{"type":"object","name":"BoxZoomTool","id":"1e0b711b-aa40-4cc5-8d13-cc8ba9091fd1","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"279333a1-5de0-4137-a398-d41000f5c94e","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"c22c8abb-22e7-4fd6-9dd9-ca1b0f4b0bfb"}],"active_drag":{"id":"b1a927e1-14eb-461a-9abb-ff92507f4403"},"active_scroll":{"id":"d05b9d44-c600-422d-b8a0-cfc16a21d709"}}},"left":[{"type":"object","name":"LinearAxis","id":"cebc8e32-6d88-42ac-8a9a-7ff71d819f6a","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"97af4fc8-6f42-4022-bef3-739ab6a6a8fd","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"000522a1-f765-493e-9fe3-c44e2075373f"},"axis_label":"body_mass_g","major_label_policy":{"type":"object","name":"AllLabels","id":"7479367b-2de1-440a-a35b-b20faf2e9a2c"}}}],"right":[{"type":"object","name":"Legend","id":"895fa423-4031-4741-9979-99e7dca3d221","attributes":{"location":[0,0],"title":"species","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"47ed77c6-38f1-4cdb-9311-ffaf61def9cb","attributes":{"label":{"type":"value","value":"Adelie"},"renderers":[{"id":"ac5edb85-3e50-4fe4-a935-cc309206888d"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"6d92c0cd-7a2b-455a-a4b3-0b3e00191c1e","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"9de11f7b-3a1a-4b65-a04d-e8f6d357fb53","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"b7d3df57-45f7-461f-b8a1-4d083be01ec4"},"axis_label":"flipper_length_mm","major_label_policy":{"type":"object","name":"AllLabels","id":"cb5994ff-be57-437e-9926-c325cd9a6d11"}}}],"center":[{"type":"object","name":"Grid","id":"c9f39fc2-40c5-4ea3-81e7-e520f08d2d35","attributes":{"axis":{"id":"6d92c0cd-7a2b-455a-a4b3-0b3e00191c1e"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"3769e8fd-2c86-422d-82b3-a37da78ce5a1","attributes":{"dimension":1,"axis":{"id":"cebc8e32-6d88-42ac-8a9a-7ff71d819f6a"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}}]}}]}},{"type":"object","name":"panel.models.comm_manager.CommManager","id":"1e8ede5f-db12-4897-81d8-7ee8776c7cba","attributes":{"plot_id":"37b483bd-505e-40ca-b373-3dd614b1254d","comm_id":"b59e30522d77473287ee438a08fa9509","client_comm_id":"960471f11182478a9a4903b1bc5dcab9"}}],"callbacks":{"type":"map"}}};
  var render_items = [{"docid":"577ee033-a386-408a-8cae-264ee6081fd7","roots":{"37b483bd-505e-40ca-b373-3dd614b1254d":"aa23c53d-f013-4ded-8fba-4abe6c1aa13f"},"root_ids":["37b483bd-505e-40ca-b373-3dd614b1254d"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined && ( root['Tabulator'] !== undefined) && ( root['Tabulator'] !== undefined))
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>




```python
islands = df["island"].unique().tolist()
sexes = df["sex"].unique().tolist()

sex_select = pn.widgets.Select(options=sexes)
island_select = pn.widgets.RadioButtonGroup(options=islands)

def create_plot(island, sex):
    return df[(df["island"] == island) & (df["sex"] == sex)].hvplot.scatter(
        y="body_mass_g",
        x="flipper_length_mm",
        by="species",
        hover_cols = "name")

interact(create_plot, island=island_select, sex=sex_select)
```








<div id='a2efd624-6e50-4d77-9d1c-a7332ec734c2'>
  <div id="f512d42f-9adf-46b3-930b-2f087aafcd01" data-root-id="a2efd624-6e50-4d77-9d1c-a7332ec734c2" style="display: contents;"></div>
</div>
<script type="application/javascript">(function(root) {
  var docs_json = {"52be93d6-2c1f-4e97-a55d-c91e6ca46dae":{"version":"3.1.1","title":"Bokeh Application","defs":[{"type":"model","name":"ReactiveHTML1"},{"type":"model","name":"FlexBox1","properties":[{"name":"align_content","kind":"Any","default":"flex-start"},{"name":"align_items","kind":"Any","default":"flex-start"},{"name":"flex_direction","kind":"Any","default":"row"},{"name":"flex_wrap","kind":"Any","default":"wrap"},{"name":"justify_content","kind":"Any","default":"flex-start"}]},{"type":"model","name":"FloatPanel1","properties":[{"name":"config","kind":"Any","default":{"type":"map"}},{"name":"contained","kind":"Any","default":true},{"name":"position","kind":"Any","default":"right-top"},{"name":"offsetx","kind":"Any","default":null},{"name":"offsety","kind":"Any","default":null},{"name":"theme","kind":"Any","default":"primary"},{"name":"status","kind":"Any","default":"normalized"}]},{"type":"model","name":"GridStack1","properties":[{"name":"mode","kind":"Any","default":"warn"},{"name":"ncols","kind":"Any","default":null},{"name":"nrows","kind":"Any","default":null},{"name":"allow_resize","kind":"Any","default":true},{"name":"allow_drag","kind":"Any","default":true},{"name":"state","kind":"Any","default":[]}]},{"type":"model","name":"drag1","properties":[{"name":"slider_width","kind":"Any","default":5},{"name":"slider_color","kind":"Any","default":"black"},{"name":"value","kind":"Any","default":50}]},{"type":"model","name":"click1","properties":[{"name":"terminal_output","kind":"Any","default":""},{"name":"debug_name","kind":"Any","default":""},{"name":"clears","kind":"Any","default":0}]},{"type":"model","name":"FastWrapper1","properties":[{"name":"object","kind":"Any","default":null},{"name":"style","kind":"Any","default":null}]},{"type":"model","name":"NotificationAreaBase1","properties":[{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0}]},{"type":"model","name":"NotificationArea1","properties":[{"name":"notifications","kind":"Any","default":[]},{"name":"position","kind":"Any","default":"bottom-right"},{"name":"_clear","kind":"Any","default":0},{"name":"types","kind":"Any","default":[{"type":"map","entries":[["type","warning"],["background","#ffc107"],["icon",{"type":"map","entries":[["className","fas fa-exclamation-triangle"],["tagName","i"],["color","white"]]}]]},{"type":"map","entries":[["type","info"],["background","#007bff"],["icon",{"type":"map","entries":[["className","fas fa-info-circle"],["tagName","i"],["color","white"]]}]]}]}]},{"type":"model","name":"Notification","properties":[{"name":"background","kind":"Any","default":null},{"name":"duration","kind":"Any","default":3000},{"name":"icon","kind":"Any","default":null},{"name":"message","kind":"Any","default":""},{"name":"notification_type","kind":"Any","default":null},{"name":"_destroyed","kind":"Any","default":false}]},{"type":"model","name":"TemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"BootstrapTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]},{"type":"model","name":"MaterialTemplateActions1","properties":[{"name":"open_modal","kind":"Any","default":0},{"name":"close_modal","kind":"Any","default":0}]}],"roots":[{"type":"object","name":"Column","id":"a2efd624-6e50-4d77-9d1c-a7332ec734c2","attributes":{"name":"Column58199","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"type":"object","name":"ImportedStyleSheet","id":"289001ca-2bbe-4102-ac62-dcfb0a576bdb","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/loading.css"}},{"type":"object","name":"ImportedStyleSheet","id":"4acd5924-7101-4a6b-86f1-e999e51523c1","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/listpanel.css"}},{"type":"object","name":"ImportedStyleSheet","id":"aaf020ad-b496-4fa3-a7c7-b1448c62a1e5","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/default.css"}},{"type":"object","name":"ImportedStyleSheet","id":"88a33f23-a7b1-4bf3-9b7c-fb6888a537d1","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/bundled/theme/native.css"}}],"margin":0,"sizing_mode":"fixed","align":"start","children":[{"type":"object","name":"Column","id":"73639cb0-a2bc-487c-a635-9771cc3322be","attributes":{"name":"Column58318","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"289001ca-2bbe-4102-ac62-dcfb0a576bdb"},{"id":"4acd5924-7101-4a6b-86f1-e999e51523c1"},{"id":"aaf020ad-b496-4fa3-a7c7-b1448c62a1e5"},{"id":"88a33f23-a7b1-4bf3-9b7c-fb6888a537d1"}],"margin":0,"align":"start","children":[{"type":"object","name":"RadioButtonGroup","id":"781842e3-2f0e-41a3-9e21-ca772bdfaa26","attributes":{"css_classes":["solid"],"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"289001ca-2bbe-4102-ac62-dcfb0a576bdb"},{"type":"object","name":"ImportedStyleSheet","id":"5e1b18c0-7f85-4418-9a7e-8ee0cb24cd60","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/button.css"}},{"id":"aaf020ad-b496-4fa3-a7c7-b1448c62a1e5"},{"id":"88a33f23-a7b1-4bf3-9b7c-fb6888a537d1"}],"margin":[5,10],"align":"start","labels":["Torgersen","Biscoe","Dream"],"active":0}},{"type":"object","name":"panel.models.widgets.CustomSelect","id":"863859bd-1d48-4f95-8c16-e92d5f0b0e13","attributes":{"stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"289001ca-2bbe-4102-ac62-dcfb0a576bdb"},{"type":"object","name":"ImportedStyleSheet","id":"f3dcfc3d-ef17-41b7-825b-bf8c123b1d7a","attributes":{"url":"https://cdn.holoviz.org/panel/1.1.0/dist/css/select.css"}},{"id":"aaf020ad-b496-4fa3-a7c7-b1448c62a1e5"},{"id":"88a33f23-a7b1-4bf3-9b7c-fb6888a537d1"}],"width":300,"min_width":300,"margin":[5,10],"align":"start","options":["male","female"],"value":"male"}}]}},{"type":"object","name":"Row","id":"eed50c0c-f87a-4136-994d-fb143f3a10c1","attributes":{"name":"Row58315","stylesheets":["\n:host(.pn-loading.pn-arc):before, .pn-loading.pn-arc:before {\n  background-image: url(\"data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHN0eWxlPSJtYXJnaW46IGF1dG87IGJhY2tncm91bmQ6IG5vbmU7IGRpc3BsYXk6IGJsb2NrOyBzaGFwZS1yZW5kZXJpbmc6IGF1dG87IiB2aWV3Qm94PSIwIDAgMTAwIDEwMCIgcHJlc2VydmVBc3BlY3RSYXRpbz0ieE1pZFlNaWQiPiAgPGNpcmNsZSBjeD0iNTAiIGN5PSI1MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzNjM2MzIiBzdHJva2Utd2lkdGg9IjEwIiByPSIzNSIgc3Ryb2tlLWRhc2hhcnJheT0iMTY0LjkzMzYxNDMxMzQ2NDE1IDU2Ljk3Nzg3MTQzNzgyMTM4Ij4gICAgPGFuaW1hdGVUcmFuc2Zvcm0gYXR0cmlidXRlTmFtZT0idHJhbnNmb3JtIiB0eXBlPSJyb3RhdGUiIHJlcGVhdENvdW50PSJpbmRlZmluaXRlIiBkdXI9IjFzIiB2YWx1ZXM9IjAgNTAgNTA7MzYwIDUwIDUwIiBrZXlUaW1lcz0iMDsxIj48L2FuaW1hdGVUcmFuc2Zvcm0+ICA8L2NpcmNsZT48L3N2Zz4=\");\n  background-size: auto calc(min(50%, 400px));\n}",{"id":"289001ca-2bbe-4102-ac62-dcfb0a576bdb"},{"id":"4acd5924-7101-4a6b-86f1-e999e51523c1"},{"id":"aaf020ad-b496-4fa3-a7c7-b1448c62a1e5"},{"id":"88a33f23-a7b1-4bf3-9b7c-fb6888a537d1"}],"margin":0,"sizing_mode":"fixed","align":"start","children":[{"type":"object","name":"Figure","id":"2977327d-d13f-463f-8ded-49b916ef1a8f","attributes":{"name":"interactive58196","width":700,"height":300,"margin":[5,10],"sizing_mode":"fixed","align":"start","x_range":{"type":"object","name":"Range1d","id":"747b54f1-0aa4-45e1-ae15-90a995efb211","attributes":{"tags":[[["flipper_length_mm","flipper_length_mm",null]]],"start":179.75714285714287,"end":211.24285714285713,"reset_start":179.75714285714287,"reset_end":211.24285714285713}},"y_range":{"type":"object","name":"Range1d","id":"4cf62b34-1c12-4b46-abb6-a46dcc7d46e3","attributes":{"tags":[[["body_mass_g","body_mass_g",null]]],"start":3187.5,"end":4837.5,"reset_start":3187.5,"reset_end":4837.5}},"x_scale":{"type":"object","name":"LinearScale","id":"69a1e2f5-a6fa-487f-a7c0-18daabad0db7"},"y_scale":{"type":"object","name":"LinearScale","id":"aa93a73a-9fa5-4e7c-8965-fa5343b6dade"},"title":{"type":"object","name":"Title","id":"768e0527-e120-450c-b050-69e8309d60cf","attributes":{"text_color":"black","text_font_size":"12pt"}},"renderers":[{"type":"object","name":"GlyphRenderer","id":"9ca68ad5-01a4-413b-8b77-306cbc2d6eb8","attributes":{"data_source":{"type":"object","name":"ColumnDataSource","id":"3a9c6daa-e1bc-4e66-a2de-54ba1a557b5d","attributes":{"selected":{"type":"object","name":"Selection","id":"d5266bf0-fe0c-4ed9-9ded-0cd0055670cb","attributes":{"indices":[],"line_indices":[]}},"selection_policy":{"type":"object","name":"UnionRenderers","id":"dfec5bef-55db-4ee0-9504-7ad280eae5a2"},"data":{"type":"map","entries":[["flipper_length_mm",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAACgZkAAAAAAAMBnQAAAAAAAYGhAAAAAAADgZ0AAAAAAAMBoQAAAAAAAoGhAAAAAAABAaEAAAAAAAMBoQAAAAAAAwGdAAAAAAACgaEAAAAAAAGBoQAAAAAAAAGdAAAAAAABgaEAAAAAAAIBoQAAAAAAAIGhAAAAAAADgaEAAAAAAAKBnQAAAAAAAwGhAAAAAAABAaUAAAAAAAOBoQAAAAAAAYGhAAAAAAABAakAAAAAAAKBoQA=="},"shape":[23],"dtype":"float64","order":"little"}],["body_mass_g",{"type":"ndarray","array":{"type":"bytes","data":"AAAAAABMrUAAAAAAAISsQAAAAAAAQ7JAAAAAAACwrUAAAAAAADCxQAAAAAAAlLFAAAAAAABosEAAAAAAAGKxQAAAAAAAeK5AAAAAAAA2sEAAAAAAAJqwQAAAAAAAeK5AAAAAAABAr0AAAAAAAFyyQAAAAAAAaLBAAAAAAAB+rUAAAAAAAPqpQAAAAAAAWKtAAAAAAABGrkAAAAAAAECvQAAAAAAAzLBAAAAAAABAr0AAAAAAAFirQA=="},"shape":[23],"dtype":"float64","order":"little"}],["name",["Michael","Thomas","Andreas","Christian","Stefan","Daniel","Peter","J\u00f6rg","Kevin","Mario","Robert","Sascha","Christoph","Stephan","Steffen","Manuel","Kai","Mike","Thorsten","Rainer","Fabian","Lars","Lukas"]],["species",["Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie","Adelie"]]]}}},"view":{"type":"object","name":"CDSView","id":"f05f8d38-c883-46b9-bfd3-4de5b44ee7d3","attributes":{"filter":{"type":"object","name":"AllIndices","id":"a854f109-db03-4d40-86bb-021d70507dd7"}}},"glyph":{"type":"object","name":"Scatter","id":"95560720-a086-4add-a6bd-5a0e76804842","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"fill_color":{"type":"value","value":"#30a2da"},"hatch_color":{"type":"value","value":"#30a2da"}}},"selection_glyph":{"type":"object","name":"Scatter","id":"8258fa02-99ba-406d-aa81-8a56b80b8def","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"angle":{"type":"value","value":0.0},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":1.0},"line_width":{"type":"value","value":1},"line_join":{"type":"value","value":"bevel"},"line_cap":{"type":"value","value":"butt"},"line_dash":{"type":"value","value":[]},"line_dash_offset":{"type":"value","value":0},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":1.0},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":1.0},"hatch_scale":{"type":"value","value":12.0},"hatch_pattern":{"type":"value","value":null},"hatch_weight":{"type":"value","value":1.0},"marker":{"type":"value","value":"circle"}}},"nonselection_glyph":{"type":"object","name":"Scatter","id":"98f2816c-7292-4793-878c-4feb557ebd2a","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.1},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.1},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.1}}},"muted_glyph":{"type":"object","name":"Scatter","id":"ced40b7c-5005-4666-b178-48c5ef2343ca","attributes":{"tags":["apply_ranges"],"x":{"type":"field","field":"flipper_length_mm"},"y":{"type":"field","field":"body_mass_g"},"size":{"type":"value","value":5.477225575051661},"line_color":{"type":"value","value":"#30a2da"},"line_alpha":{"type":"value","value":0.2},"fill_color":{"type":"value","value":"#30a2da"},"fill_alpha":{"type":"value","value":0.2},"hatch_color":{"type":"value","value":"#30a2da"},"hatch_alpha":{"type":"value","value":0.2}}}}}],"toolbar":{"type":"object","name":"Toolbar","id":"a93ba77a-3875-4dc2-acaf-d484e443feef","attributes":{"tools":[{"type":"object","name":"HoverTool","id":"c4538b66-ad11-49b9-ae6c-2aa1dbc7c238","attributes":{"tags":["hv_created"],"renderers":[{"id":"9ca68ad5-01a4-413b-8b77-306cbc2d6eb8"}],"tooltips":[["species","@{species}"],["flipper_length_mm","@{flipper_length_mm}"],["body_mass_g","@{body_mass_g}"],["name","@{name}"]]}},{"type":"object","name":"SaveTool","id":"bb09eb5a-f7d4-47df-b1c8-f29f1a05c839"},{"type":"object","name":"PanTool","id":"73477d60-3613-4d1c-8e6b-b1e07483e38c"},{"type":"object","name":"WheelZoomTool","id":"e33ce4d6-40ee-40fe-9a4c-03e4a0eec075"},{"type":"object","name":"BoxZoomTool","id":"9cee68e7-59c9-466d-a663-a9c66032be0e","attributes":{"overlay":{"type":"object","name":"BoxAnnotation","id":"317dd157-48fa-4fc3-b4aa-767947e1deb8","attributes":{"syncable":false,"level":"overlay","visible":false,"left_units":"canvas","right_units":"canvas","bottom_units":"canvas","top_units":"canvas","line_color":"black","line_alpha":1.0,"line_width":2,"line_dash":[4,4],"fill_color":"lightgrey","fill_alpha":0.5}}}},{"type":"object","name":"ResetTool","id":"5ec2642e-c9a7-4b58-a4da-1fd7adad7820"}],"active_drag":{"id":"73477d60-3613-4d1c-8e6b-b1e07483e38c"},"active_scroll":{"id":"e33ce4d6-40ee-40fe-9a4c-03e4a0eec075"}}},"left":[{"type":"object","name":"LinearAxis","id":"1b97de3f-c814-4e45-9473-0021b7341664","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"c73bca01-833d-4cb4-b153-704bd31765f8","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"979d569f-63ff-4e2e-bdc1-48ce6b56cf9e"},"axis_label":"body_mass_g","major_label_policy":{"type":"object","name":"AllLabels","id":"43786ac4-e399-4c24-85ee-8966e27e2300"}}}],"right":[{"type":"object","name":"Legend","id":"a7ff47ee-852f-4adb-9672-3502f17b4891","attributes":{"location":[0,0],"title":"species","click_policy":"mute","items":[{"type":"object","name":"LegendItem","id":"fda3e041-f90e-4bbb-beb1-fde0e960e389","attributes":{"label":{"type":"value","value":"Adelie"},"renderers":[{"id":"9ca68ad5-01a4-413b-8b77-306cbc2d6eb8"}]}}]}}],"below":[{"type":"object","name":"LinearAxis","id":"07238eee-b6c0-458d-b9f9-afa78fcff7db","attributes":{"ticker":{"type":"object","name":"BasicTicker","id":"84a30ef3-4c07-4610-b3c4-3797c6897f26","attributes":{"mantissas":[1,2,5]}},"formatter":{"type":"object","name":"BasicTickFormatter","id":"e9ac8afb-4463-437c-bbeb-d8842b484d5d"},"axis_label":"flipper_length_mm","major_label_policy":{"type":"object","name":"AllLabels","id":"cbddf922-e3b0-4efa-bd01-16dac8626524"}}}],"center":[{"type":"object","name":"Grid","id":"a25a6dbe-cd71-4ff1-99bf-5eea6ec56229","attributes":{"axis":{"id":"07238eee-b6c0-458d-b9f9-afa78fcff7db"},"grid_line_color":null}},{"type":"object","name":"Grid","id":"b7cd1b3f-cc3d-4b11-b134-333076dd0778","attributes":{"dimension":1,"axis":{"id":"1b97de3f-c814-4e45-9473-0021b7341664"},"grid_line_color":null}}],"min_border_top":10,"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"output_backend":"webgl"}}]}}]}},{"type":"object","name":"panel.models.comm_manager.CommManager","id":"c647051a-78fe-490b-831e-923537ee74ba","attributes":{"plot_id":"a2efd624-6e50-4d77-9d1c-a7332ec734c2","comm_id":"b877bd5b90684cefa970565636928e09","client_comm_id":"87bc71d147d0472492cf7a11e16664d6"}}],"callbacks":{"type":"map"}}};
  var render_items = [{"docid":"52be93d6-2c1f-4e97-a55d-c91e6ca46dae","roots":{"a2efd624-6e50-4d77-9d1c-a7332ec734c2":"f512d42f-9adf-46b3-930b-2f087aafcd01"},"root_ids":["a2efd624-6e50-4d77-9d1c-a7332ec734c2"]}];
  var docs = Object.values(docs_json)
  if (!docs) {
    return
  }
  const py_version = docs[0].version.replace('rc', '-rc.')
  const is_dev = py_version.indexOf("+") !== -1 || py_version.indexOf("-") !== -1
  function embed_document(root) {
    var Bokeh = get_bokeh(root)
    Bokeh.embed.embed_items_notebook(docs_json, render_items);
    for (const render_item of render_items) {
      for (const root_id of render_item.root_ids) {
	const id_el = document.getElementById(root_id)
	if (id_el.children.length && (id_el.children[0].className === 'bk-root')) {
	  const root_el = id_el.children[0]
	  root_el.id = root_el.id + '-rendered'
	}
      }
    }
  }
  function get_bokeh(root) {
    if (root.Bokeh === undefined) {
      return null
    } else if (root.Bokeh.version !== py_version && !is_dev) {
      if (root.Bokeh.versions === undefined || !root.Bokeh.versions.has(py_version)) {
	return null
      }
      return root.Bokeh.versions.get(py_version);
    } else if (root.Bokeh.version === py_version) {
      return root.Bokeh
    }
    return null
  }
  function is_loaded(root) {
    var Bokeh = get_bokeh(root)
    return (Bokeh != null && Bokeh.Panel !== undefined && ( root['Tabulator'] !== undefined) && ( root['Tabulator'] !== undefined))
  }
  if (is_loaded(root)) {
    embed_document(root);
  } else {
    var attempts = 0;
    var timer = setInterval(function(root) {
      if (is_loaded(root)) {
        clearInterval(timer);
        embed_document(root);
      } else if (document.readyState == "complete") {
        attempts++;
        if (attempts > 200) {
          clearInterval(timer);
	  var Bokeh = get_bokeh(root)
	  if (Bokeh == null || Bokeh.Panel == null) {
            console.warn("Panel: ERROR: Unable to run Panel code because Bokeh or Panel library is missing");
	  } else {
	    console.warn("Panel: WARNING: Attempting to render but not all required libraries could be resolved.")
	    embed_document(root)
	  }
        }
      }
    }, 25, root)
  }
})(window);</script>



## Splitting into training and test sets


```python
from sklearn.model_selection import train_test_split
from pandas import get_dummies

y = df["species"].values
penguins_x = pd.get_dummies(df.drop("species", axis = 1))
x = penguins_x.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .1, random_state = 187)
```

## Defining a model


```python
from sklearn.ensemble import RandomForestClassifier

steps = [("rf", RandomForestClassifier(n_estimators = 100))]
```

## Setting up a pipeline


```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps = steps)

rf_pipeline_model = pipeline.fit(x_train, y_train)

rf_pred_tbl = pd.DataFrame(
{"true": y_test,
"pred": rf_pipeline_model.predict(x_test)})

rf_pred_tbl
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chinstrap</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gentoo</td>
      <td>Gentoo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gentoo</td>
      <td>Gentoo</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>Chinstrap</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Chinstrap</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Chinstrap</td>
      <td>Chinstrap</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 2 columns</p>
</div>



<h2>Evaluation</h2>


```python
from sklearn.metrics import classification_report

classification_report = classification_report(
y_true = rf_pred_tbl.true,
y_pred = rf_pred_tbl.pred)

print(classification_report)
```

                  precision    recall  f1-score   support
    
          Adelie       0.72      1.00      0.84       126
       Chinstrap       1.00      0.23      0.38        64
          Gentoo       1.00      1.00      1.00       110
    
        accuracy                           0.84       300
       macro avg       0.91      0.74      0.74       300
    weighted avg       0.88      0.84      0.80       300
    
    


```python

```
