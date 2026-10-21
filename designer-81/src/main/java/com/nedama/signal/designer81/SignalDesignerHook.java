package com.nedama.signal.designer81;

import com.inductiveautomation.ignition.common.script.ScriptManager;
import com.inductiveautomation.ignition.designer.model.AbstractDesignerModuleHook;
import com.nedama.signal.common81.SignalRegistration;

/** Designer scope: script console and autocomplete hints. */
public class SignalDesignerHook extends AbstractDesignerModuleHook {

    @Override
    public void initializeScriptManager(ScriptManager manager) {
        super.initializeScriptManager(manager);
        SignalRegistration.registerScripts(manager);
    }
}
