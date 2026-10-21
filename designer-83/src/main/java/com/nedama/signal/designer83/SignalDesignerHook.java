package com.nedama.signal.designer83;

import com.inductiveautomation.ignition.common.script.ScriptManager;
import com.inductiveautomation.ignition.designer.model.AbstractDesignerModuleHook;
import com.nedama.signal.common83.SignalRegistration;

/** Designer scope: script console and autocomplete hints. */
public class SignalDesignerHook extends AbstractDesignerModuleHook {

    @Override
    public void initializeScriptManager(ScriptManager manager) {
        super.initializeScriptManager(manager);
        SignalRegistration.registerScripts(manager);
    }
}
